import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Optional, Union
import torch.nn.modules.transformer as F_transformer
from .layer_history import CreateLayerHistory


class ODETransformerEncoder(nn.TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, norm=None, calculate_num=2, rk_type="learnable", history_args=None):
        super().__init__(encoder_layer, num_layers, norm)
        self.calculate_num = calculate_num
        self.rk_type = rk_type
       
        self.history = CreateLayerHistory(args=history_args, is_encoder=True)
        print(f"validation of all arguments in ODETransformerEncoder: calculate_num={self.calculate_num}, rk_type={self.rk_type}, history_args={history_args}")
        # Get hidden dimension from encoder layer
        hidden_dim = encoder_layer.linear2.out_features if hasattr(encoder_layer, 'linear2') else encoder_layer.self_attn.embed_dim
        
    
        if self.calculate_num == 2 and self.rk_type == "learnable":
            # RK2-learnable
            self.gate_linear = nn.Linear(hidden_dim * 2, hidden_dim)
            print(self.gate_linear)
            print(self.gate_linear.weight)
            print(self.gate_linear.weight.size())

        elif self.rk_type == "initialization":
            self.alpha = nn.Parameter(torch.Tensor(self.calculate_num))
            self.alpha.data.fill_(1)

        elif self.calculate_num == 4 and self.rk_type == "learnable":
            self.alpha = nn.Parameter(torch.Tensor(self.calculate_num))
            self.alpha.data.fill_(1.0/self.calculate_num)
    
    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: Optional[bool] = None,
    ) -> torch.Tensor:
        r"""Pass the input through the encoder layers using Runge-Kutta methods."""
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype,
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )
        if self.history is not None:
            self.history.clean()
        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ""
        str_first_layer = "self.layers[0]"
        batch_first = first_layer.self_attn.batch_first
        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        if not is_fastpath_enabled:
            why_not_sparsity_fast_path = (
                "torch.backends.mha.get_fastpath_enabled() was not True"
            )
        elif not hasattr(self, "use_nested_tensor"):
            why_not_sparsity_fast_path = "use_nested_tensor attribute not present"
        elif not self.use_nested_tensor:
            why_not_sparsity_fast_path = (
                "self.use_nested_tensor (set in init) was not True"
            )
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = (
                f"input not batched; expected src.dim() of 3 but got {src.dim()}"
            )
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        elif (
            (not hasattr(self, "mask_check")) or self.mask_check
        ) and not torch._nested_tensor_from_mask_left_aligned(
            src, src_key_padding_mask.logical_not()
        ):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = (
                "src_key_padding_mask and mask were both supplied"
            )
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )
            _supported_device_type = [
                "cpu",
                "cuda",
                torch.utils.backend_registration._privateuse1_backend_name,
            ]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif src.device.type not in _supported_device_type:
                why_not_sparsity_fast_path = (
                    f"src device is neither one of {_supported_device_type}"
                )
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad"
                )

            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(
                    output, src_key_padding_mask.logical_not(), mask_check=False
                )
                src_key_padding_mask_for_layers = None
        
        seq_len = F_transformer._get_seq_len(src, batch_first)
        is_causal = F_transformer._detect_is_causal_mask(mask, is_causal, seq_len)

        # Replace original layer iteration with Runge-Kutta implementation
        x = output
        # History ilk durumu eklemeden pop yapmayın
        if self.history is not None:
            self.history.add(x)
        for layer in self.layers:
            if self.history is not None:
                x = self.history.pop()
                
            runge_kutta_list = []
            residual = x
            
            for step_size in range(self.calculate_num):
                # Call the layer with appropriate parameters
                step_output = layer(
                    x,
                    src_mask=mask,
                    is_causal=is_causal,
                    src_key_padding_mask=src_key_padding_mask_for_layers,
                )
                
                runge_kutta_list.append(step_output)
                
                if self.calculate_num == 4:
                    if step_size == 0 or step_size == 1:
                        x = residual + 1 / 2 * step_output
                    elif step_size == 2:
                        x = residual + step_output
                elif self.calculate_num == 3:
                    if step_size == 0:
                        x = residual + 1 / 2 * step_output
                    elif step_size == 1:
                        x = residual - runge_kutta_list[0] + 2 * step_output
                elif self.calculate_num == 2:
                    x = residual + step_output
            
            if self.calculate_num == 4:
                # RK4-block
                if self.rk_type == "standard":
                    x = residual + 1/6 * (runge_kutta_list[0] + 2*runge_kutta_list[1] + 2*runge_kutta_list[2] + runge_kutta_list[3])
                if self.rk_type == "initialization" or self.rk_type == "learnable":
                    x = residual + self.alpha[0] * runge_kutta_list[0] + self.alpha[1] * runge_kutta_list[1] + self.alpha[2] * runge_kutta_list[2] + self.alpha[3] * runge_kutta_list[3] 
            elif self.calculate_num == 3:
                # RK3-block
                x = residual + 1/6 * (runge_kutta_list[0] + 4*runge_kutta_list[1] + runge_kutta_list[2])
            elif self.calculate_num == 2:
                # learnable coefficients for RK2-block with gated
                if self.rk_type == "learnable":
                    alpha = torch.sigmoid(self.gate_linear(torch.cat((runge_kutta_list[0], runge_kutta_list[1]), dim=-1)))
                    x = residual + alpha * runge_kutta_list[0] + (1 - alpha) * runge_kutta_list[1]
                elif self.rk_type == "standard":
                    # RK2-block
                    x = residual + 1/2 * (runge_kutta_list[0] + runge_kutta_list[1])
                elif self.rk_type == "initialization":
                    # learnable coefficients with initialized 1
                    x = residual + self.alpha[0] * runge_kutta_list[0] + self.alpha[1] * runge_kutta_list[1]
            elif self.calculate_num == 1:
                # Euler-block
                if self.rk_type == "residual":
                    x = residual + runge_kutta_list[0]
                else:
                    x = runge_kutta_list[0]
            else:
                raise ValueError("Invalid calculate_num!")
            
            if self.history is not None:
                self.history.add(x)
        # Final processing
        if self.history is not None:
            x = self.history.pop()
            
        output = x
        
        if convert_to_nested:
            output = output.to_padded_tensor(0.0, src.size())
        
        if self.norm is not None:
            output = self.norm(output)
        
        return output
