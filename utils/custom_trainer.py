import torch
from torch.utils.data import DataLoader
from typing import Optional
from transformers import Trainer
from transformers.utils import is_torch_xla_available
from transformers.trainer_utils import SaveStrategy, EvalLoopOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

class CustomTrainer(Trainer):
    
    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
    ):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()
            
            # Custom logging for prefix loss
            assert model.training
            
            if hasattr(model, "softmax_ce_loss"):
                logs["softmax_ce_loss"] = model.softmax_ce_loss.item()
            if hasattr(model, "weighted_prefix_loss"):
                logs["weighted_prefix_loss"] = model.weighted_prefix_loss.item()
            if hasattr(model, "total_loss"):
                logs["total_loss"] = model.total_loss.item()

            if hasattr(model, "prefix_losses"):
                for i, l in enumerate(model.prefix_losses):
                    logs[f"prefix_loss_{i+1}"] = l.item() if l is not None and isinstance(l, torch.Tensor) else None
                    
            if hasattr(model, "prefix_avg_auc"):
                for i in range(len(model.prefix_avg_auc)):
                    logs[f"prefix_avg_auc_{i+1}"] = model.prefix_avg_auc[i] if model.prefix_avg_auc[i] is not None else None
                    logs[f"prefix_avg_mrr_{i+1}"] = model.prefix_avg_mrr[i] if model.prefix_avg_mrr[i] is not None else None
            
            if hasattr(model, "adaptive_weights"):
                for i, w in enumerate(model.adaptive_weights):
                    logs[f"adaptive_weight_{i+1}"] = w if w is not None else None

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
    
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        output = super().evaluation_loop(
            dataloader=dataloader,
            description=description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        if hasattr(self.model, "softmax_ce_loss"):
            output.metrics[f"{metric_key_prefix}_softmax_ce_loss"] = self.model.softmax_ce_loss.item()
        if hasattr(self.model, "weighted_prefix_loss"):
            output.metrics[f"{metric_key_prefix}_weighted_prefix_loss"] = self.model.weighted_prefix_loss.item()
        if hasattr(self.model, "total_loss"):
            output.metrics[f"{metric_key_prefix}_total_loss"] = self.model.total_loss.item()

        if hasattr(self.model, "prefix_losses"):
            for i, l in enumerate(self.model.prefix_losses):
                output.metrics.update({
                    f"{metric_key_prefix}_prefix_loss_{i+1}": l.item() if l is not None and isinstance(l, torch.Tensor) else None,
                })
        if hasattr(self.model, "prefix_avg_auc"):
            for i in range(len(self.model.prefix_avg_auc)):
                output.metrics.update({
                    f"{metric_key_prefix}_prefix_avg_auc_{i+1}": self.model.prefix_avg_auc[i] if self.model.prefix_avg_auc[i] is not None else None,
                    f"{metric_key_prefix}_prefix_avg_mrr_{i+1}": self.model.prefix_avg_mrr[i] if self.model.prefix_avg_mrr[i] is not None else None,
                })
        if hasattr(self.model, "adaptive_weights"):
            for i, w in enumerate(self.model.adaptive_weights):
                output.metrics.update({
                    f"{metric_key_prefix}_adaptive_weight_{i+1}": w if w is not None else None
                })
        return output
        
