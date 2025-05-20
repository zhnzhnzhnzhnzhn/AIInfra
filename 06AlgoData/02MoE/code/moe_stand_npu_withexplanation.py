import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu  # 华为昇腾NPU专用库
from torch_npu.contrib import transfer_to_npu  # NPU数据迁移工具

# 专家网络定义
class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 全连接层：input_dim -> hidden_dim
            nn.GELU(),                         # GELU激活函数（比ReLU更平滑）
            nn.Linear(hidden_dim, output_dim)  # 全连接层：hidden_dim -> output_dim
        )
        
    def forward(self, x):
        return self.net(x)  # 顺序执行定义的三层结构

# 混合专家模型核心类
class MoE(nn.Module):
    def __init__(self, input_dim, num_experts, top_k, expert_capacity, hidden_dim, output_dim):
        super().__init__()
        # 关键参数初始化
        self.num_experts = num_experts    # 专家总数（例如8）
        self.top_k = top_k                # 每个样本选择的专家数（例如2）
        self.expert_capacity = expert_capacity  # 单个专家最大处理样本数（例如32）
        
        # 门控网络（路由网络）
        self.gate = nn.Linear(input_dim, num_experts)  # 输入维度到专家数量的线性层
        
        # 专家集合初始化
        self.experts = nn.ModuleList(
            [Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)]
        )  # 创建num_experts个相同结构的专家
        
    def forward(self, x):
        batch_size, input_dim = x.shape  # 获取输入维度 [batch_size, input_dim]
        device = x.device  # 确保计算设备一致
        
        # 路由计算（核心逻辑）
        logits = self.gate(x)  # 计算原始门控分数 [batch_size, num_experts]
        probs = torch.softmax(logits, dim=-1)  # 转换为概率分布
        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)  # 选择top_k专家
        
        # 辅助损失计算（仅在训练时激活）
        if self.training:
            # 重要性损失（专家利用率均衡）
            importance = probs.sum(0)  # 每个专家的总概率 [num_experts]
            importance_loss = torch.var(importance) / (self.num_experts ** 2)  # 方差归一化
            
            # 负载均衡损失（样本分配均衡）
            mask = torch.zeros_like(probs, dtype=torch.bool)  # 创建布尔掩码
            mask.scatter_(1, topk_indices, True)  # 将top_k位置标记为True
            routing_probs = probs * mask  # 仅保留激活专家的概率
            expert_usage = mask.float().mean(0)  # 各专家的使用率
            routing_weights = routing_probs.mean(0)  # 各专家的平均权重
            load_balance_loss = self.num_experts * (expert_usage * routing_weights).sum()
            
            aux_loss = importance_loss + load_balance_loss  # 综合辅助损失
        else:
            aux_loss = 0.0  # 推理时不计算辅助损失

        # 专家分配逻辑
        flat_indices = topk_indices.view(-1)  # 展平为[batch_size*top_k]
        flat_probs = topk_probs.view(-1)      # 对应概率展平
        sample_indices = torch.arange(batch_size, device=device)[:, None]\
                            .expand(-1, self.top_k).flatten()  # 生成样本索引[0,0,1,1,...]用于配对

        # 初始化输出张量
        outputs = torch.zeros(
            batch_size, 
            self.experts[0].net[-1].out_features,  # 取第一个专家的输出维度
            device=device
        )

        # 遍历所有专家进行并行计算
        for expert_idx in range(self.num_experts):
            # 获取分配给当前专家的样本
            expert_mask = flat_indices == expert_idx  # 布尔掩码筛选
            expert_samples = sample_indices[expert_mask]  # 对应的样本索引
            expert_weights = flat_probs[expert_mask]       # 对应的路由权重

            # 容量控制机制
            if len(expert_samples) > self.expert_capacity:
                # 截断超过容量的样本（按顺序保留前N个）
                expert_samples = expert_samples[:self.expert_capacity]
                expert_weights = expert_weights[:self.expert_capacity]

            if len(expert_samples) == 0:
                continue  # 无样本分配给该专家时跳过

            # 专家计算过程
            expert_input = x[expert_samples]  # 提取对应样本的输入
            expert_output = self.experts[expert_idx](expert_input)  # 专家前向计算
            weighted_output = expert_output * expert_weights.unsqueeze(-1)  # 权重广播相乘
            
            # 累加输出到最终结果
            outputs.index_add_(
                0,  # 按样本维度累加
                expert_samples, 
                weighted_output
            )

        return outputs, aux_loss  # 返回输出和辅助损失

# 测试代码解析
if __name__ == "__main__":
    # 参数设置
    input_dim = 128      # 输入特征维度
    output_dim = 256     # 输出特征维度
    num_experts = 8      # 专家数量
    top_k = 2            # 每个样本选择2个专家
    expert_capacity = 32 # 单个专家最多处理32个样本
    hidden_dim = 512     # 专家网络隐藏层维度
    batch_size = 64      # 批大小

    # 设备配置（优先使用NPU）
    device = torch.device("npu:4" if torch.npu.is_available() else "cpu")
    moe = MoE(input_dim, num_experts, top_k, expert_capacity, hidden_dim, output_dim).to(device)
    x = torch.randn(batch_size, input_dim).to(device)  # 生成测试数据

    # NPU性能分析配置
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
        msprof_tx=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        l2_cache=False,
        op_attr=False,
        data_simplification=False,
        record_op_args=False,
        gc_detect_threshold=None
    )

    # 性能分析上下文
    with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU
            ],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./moe_stand_npu_result"),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_modules=False,
            with_flops=False,
            experimental_config=experimental_config) as prof:
        # 训练模式循环
        for _ in range(10):
            moe.train()  # 设置训练模式（启用辅助损失）
            output, loss = moe(x)  # 前向传播
            print(f"Using device: {x.device}")  # 打印计算设备
            print(f"Training output shape: {output.shape}")      # 预期输出 torch.Size([64, 256])
            print(f"Training auxiliary loss: {loss.item():.4f}") # 打印辅助损失值
            prof.step()  # 记录性能分析步骤

    print("=" * 80)

    # 推理模式测试
    moe.eval()  # 设置评估模式（关闭辅助损失）
    output, _ = moe(x)  # 仅获取输出
    print(f"Eval output shape: {output.shape}")  # 验证输出形状一致性
