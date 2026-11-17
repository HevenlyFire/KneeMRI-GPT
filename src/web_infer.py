import os
import torch

from model import TripleMRNet
from submit_predictions import get_study

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DEFAULT_ABNORMAL_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "runs", "abnormal_triple_alex", "val0.1758_train0.0854_epoch24"
)
DEFAULT_ACL_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "runs", "acl_triple_alex", "val0.1370_train0.0717_epoch22"
)
DEFAULT_MENISCUS_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "runs", "meniscus_triple_alex", "val0.2641_train0.1819_epoch10"
)
DEFAULT_BACKBONE = "alexnet"


class MRDiagnosisService:
    def __init__(
        self,
        abnormal_model_path=DEFAULT_ABNORMAL_MODEL_PATH,
        acl_model_path=DEFAULT_ACL_MODEL_PATH,
        meniscus_model_path=DEFAULT_MENISCUS_MODEL_PATH,
        backbone=DEFAULT_BACKBONE,
    ):
        self.abnormal_model_path = abnormal_model_path
        self.acl_model_path = acl_model_path
        self.meniscus_model_path = meniscus_model_path
        self.backbone = backbone
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.abnormal_model = None
        self.acl_model = None
        self.meniscus_model = None

    def _load_model(self, model_path):
        model = TripleMRNet(backbone=self.backbone, training=False)
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def _ensure_models_loaded(self):
        if self.abnormal_model is None:
            self.abnormal_model = self._load_model(self.abnormal_model_path)
        if self.acl_model is None:
            self.acl_model = self._load_model(self.acl_model_path)
        if self.meniscus_model is None:
            self.meniscus_model = self._load_model(self.meniscus_model_path)

    def _predict_single_task(self, model, tensors, abnormality_prior=None):
        with torch.no_grad():
            vol_axial = tensors["axial"].to(self.device)
            vol_sagit = tensors["sagit"].to(self.device)
            vol_coron = tensors["coron"].to(self.device)

            logit = model.forward(vol_axial, vol_sagit, vol_coron)
            pred = torch.sigmoid(logit)
            pred_value = float(pred.detach().cpu().numpy()[0][0])

            if abnormality_prior is not None:
                pred_value = float(pred_value * abnormality_prior)
            return pred_value

    def predict_three_tasks(self, axial_path, sagittal_path, coronal_path):
        self._ensure_models_loaded()
        study_tensors = get_study(axial_path, sagittal_path, coronal_path)

        abnormal_prob = self._predict_single_task(self.abnormal_model, study_tensors)
        acl_prob = self._predict_single_task(
            self.acl_model, study_tensors, abnormality_prior=abnormal_prob
        )
        meniscus_prob = self._predict_single_task(
            self.meniscus_model, study_tensors, abnormality_prior=abnormal_prob
        )

        return {
            "abnormal": abnormal_prob,
            "acl": acl_prob,
            "meniscus": meniscus_prob,
        }
