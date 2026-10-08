# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLOv8 dual detection and human-pose training and inference entry point."""

from copy import copy

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect import DetectionPredictor, DetectionValidator
from ultralytics.models.yolo.model import YOLO
from ultralytics.models.yolo.pose import PoseTrainer, PoseValidator
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK, nms, ops


def _split_predictions(preds):
    """Return detection and pose predictions from native or exported model outputs."""
    return (preds["detect"], preds["pose"]) if isinstance(preds, dict) else preds


class DualTaskModel(DetectionModel):
    """DetectionModel whose final layer emits independent detection and human-pose predictions."""

    def __init__(self, cfg, ch=3, nc=None, verbose=True):
        """Build a dual-head model and expose its pose metadata."""
        super().__init__(cfg, ch, nc, verbose)
        self.kpt_shape = self.model[-1].kpt_shape


class DualTaskValidator(PoseValidator):
    """Validate the human-pose branch of a dual-task model."""

    def __call__(self, trainer=None, model=None, **kwargs):
        """Evaluate pose and detection independently on the same pose-format dataloader."""
        pose_stats = super().__call__(trainer=trainer, model=model, **kwargs)
        detection = DualDetectionValidator(self.dataloader, self.save_dir, copy(self.args), self.callbacks)
        det_stats = detection(trainer=trainer, model=model, **kwargs)
        self.metrics.det = detection.metrics
        if pose_stats is not None and det_stats is not None:
            pose_stats.update(
                {f"detect/{key}": value for key, value in det_stats.items() if key.startswith("metrics/")}
            )
        return pose_stats

    def init_metrics(self, model):
        """Use the pose branch's single class and its configured detection-class mapping."""
        super().init_metrics(model)
        native = model.model if getattr(model, "format", None) == "pt" else model
        metadata = getattr(native, "metadata", {})
        self.person_class = int(getattr(native, "yaml", {}).get("person_class", metadata.get("person_class", 0)))
        self.names = {0: "person"}
        self.nc = 1
        self.metrics.names = self.names

    def preprocess(self, batch):
        """Move the complete batch to the validation device for the joint loss."""
        return super().preprocess(batch)

    def _prepare_batch(self, si, batch):
        """Filter non-human labels only while preparing pose metrics."""
        person = batch["cls"].view(-1).long() == self.person_class
        batch = batch.copy()
        for name in ("cls", "bboxes", "batch_idx", "keypoints"):
            batch[name] = batch[name][person]
        batch["cls"].zero_()
        return super()._prepare_batch(si, batch)

    def postprocess(self, preds):
        """Run pose NMS on the pose branch output."""
        return super().postprocess(_split_predictions(preds)[1])


class DualDetectionValidator(DetectionValidator):
    """Evaluate the detection branch of a dual-task model."""

    def postprocess(self, preds):
        """Run detection NMS on the detection branch output."""
        return super().postprocess(_split_predictions(preds)[0])


class DualTaskTrainer(PoseTrainer):
    """Train both heads from one pose-format dataset with human and non-human labels."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Build the dual model using the detection dataset class count."""
        model = self.set_model_names_for_load(
            DualTaskModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        )
        if list(model.kpt_shape) != list(self.data["kpt_shape"]):
            raise ValueError(f"Dataset kpt_shape={self.data['kpt_shape']} differs from model {model.kpt_shape}.")
        if model.model[-1].nc_pose != 1:
            raise ValueError("Dual-task human-pose training requires nc_pose=1.")
        if not 0 <= model.yaml.get("person_class", 0) < self.data["nc"]:
            raise ValueError("person_class must be a valid detection class index.")
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a pose-branch validator for checkpoint selection."""
        return DualTaskValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )


class DualTaskPredictor(DetectionPredictor):
    """Return normal detection Results with an independent pose Results in ``result.pose``."""

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Apply NMS independently to detection and pose branch outputs."""
        pred_detect, pred_pose = _split_predictions(preds)
        detected = super().postprocess(pred_detect, img, orig_imgs, **kwargs)
        posed = nms.non_max_suppression(pred_pose, self.args.conf, self.args.iou, max_det=self.args.max_det, nc=1)
        kpt_shape = getattr(self.model, "kpt_shape", None)
        if kpt_shape is None:
            pose_tensor = pred_pose[0] if isinstance(pred_pose, (tuple, list)) else pred_pose
            kpt_dims = 3 if (pose_tensor.shape[1] - 5) % 3 == 0 else 2
            kpt_shape = ((pose_tensor.shape[1] - 5) // kpt_dims, kpt_dims)
        for result, pred in zip(detected, posed):
            boxes = pred[:, :6].clone()
            boxes[:, :4] = ops.scale_boxes(img.shape[2:], boxes[:, :4], result.orig_img.shape)
            kpts = pred[:, 6:].view(-1, *kpt_shape)
            kpts = ops.scale_coords(img.shape[2:], kpts, result.orig_img.shape)
            result.pose = Results(result.orig_img, path=result.path, names={0: "person"}, boxes=boxes, keypoints=kpts)
        return detected


class DualTaskYOLO(YOLO):
    """YOLO facade for independent detection and pose heads over one YOLOv8 backbone and neck."""

    def __init__(self, model="yolo26n.pt", task="pose", verbose=False):
        """Initialize a dual-task model as a pose task so exports retain keypoint metadata."""
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Use the dual model, pose-format trainer, pose validator, and dual predictor for detection tasks."""
        task_map = super().task_map
        dual = {
            "model": DualTaskModel,
            "trainer": DualTaskTrainer,
            "validator": DualTaskValidator,
            "predictor": DualTaskPredictor,
        }
        task_map["detect"] = dual
        task_map["pose"] = dual
        return task_map
