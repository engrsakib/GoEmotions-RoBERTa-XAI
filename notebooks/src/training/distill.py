"""Knowledge distillation from multi-label teacher to single-label student."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import Trainer


class DistillationTrainer(Trainer):
    def __init__(
        self,
        *args,
        teacher_model=None,
        distill_alpha: float = 0.5,
        distill_temperature: float = 2.0,
        class_weights: torch.Tensor | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.distill_alpha = distill_alpha
        self.distill_temperature = distill_temperature
        self.class_weights = class_weights
        if teacher_model is not None:
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        student_logits = outputs.logits

        ce_loss = F.cross_entropy(student_logits, labels, weight=self.class_weights)

        if self.teacher_model is None:
            loss = ce_loss
        else:
            with torch.no_grad():
                teacher_logits = self.teacher_model(**inputs).logits
            # Teacher may be multi-label sigmoid; convert to soft 7-class distribution
            teacher_probs = torch.sigmoid(teacher_logits)
            teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            student_log_probs = F.log_softmax(
                student_logits / self.distill_temperature, dim=-1
            )
            kd_loss = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction="batchmean",
            ) * (self.distill_temperature ** 2)

            loss = self.distill_alpha * kd_loss + (1 - self.distill_alpha) * ce_loss

        return (loss, outputs) if return_outputs else loss


def load_teacher_model(model_path: str, num_labels: int = 7):
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )
    model.eval()
    return model
