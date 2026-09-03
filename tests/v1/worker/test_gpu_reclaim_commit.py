# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SimpleNamespace：
# Python 标准库提供的一个非常轻量的“临时对象”。
#
# 例如：
# runner = SimpleNamespace(
#     req_states=req_states,
#     block_tables=block_tables,
# )
#
# 之后就可以：
# runner.req_states
# runner.block_tables
#
# 这里不用真正初始化一个完整 GPUModelRunner，
# 因为 _commit_reclaim_transition() 实际只依赖：
#   self.req_states
#   self.block_tables
#
# 所以测试人为造一个“最小 self”即可。
from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.core.sched.output import CachedRequestData, ReclaimTransitionData

# Worker-side physical block-table。
#
# 注意：
# 它不是 Scheduler 的 allocator。
#
# Scheduler:
#   KVCacheManager / BlockPool
#   决定 physical block ID 属于谁。
#
# Worker:
#   BlockTables
#   保存 execution 时：
#       physical block-table column
#           ↓
#       physical block ID
#   的映射。
from vllm.v1.worker.gpu.block_table import BlockTables

# S1 中已经修改的正常 execution progress update。
#
# normal forward 完成后：
#
# logical num_computed_tokens += actual_delta
# physical effective_kv_len    += actual_delta
#
# 测试 reclaim 后：
#   L=94, E=62
#
# 再正常执行一个 token：
#   L=95, E=63
from vllm.v1.worker.gpu.input_batch import post_update

# production code 中：
#
# GPUModelRunner._commit_reclaim_transition()
#
# 就定义在这个 class 中。
#
# 测试不会真的初始化完整 GPUModelRunner，
# 但会直接调用这个真实 production method。
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

# Worker-side request persistent execution state。
#
# 其中包含：
#   num_computed_tokens       logical progress
#   effective_kv_len          physical KV progress
#   last_sampled_tokens
#   all_token_ids
#   ...
from vllm.v1.worker.gpu.states import RequestState


# ============================================================
# 整个 test file 都要求 CUDA。
#
# pytestmark:
# 给当前文件中的所有 tests 添加同一个 pytest marker。
#
# skipif(condition, reason):
# 如果 condition=True，则测试 skip。
#
# 这里：
#   如果当前不是 CUDA platform
#   就全部 skip。
#
# 所以前面曾经看到：
#   25 skipped
#
# 就是这个机制。
# ============================================================

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="requires CUDA",
)


# ============================================================
# Test Fixture：
# 构造一个“已经处于 RUNNING 状态”的最小 Worker 世界。
#
# 这个函数非常重要。
#
# 它不是在模拟：
#
# Scheduler
#   ↓
# add_requests()
#   ↓
# Worker
#
# 整条完整链路。
#
# 而是直接人为建立：
#
# “假设 add_requests 已经发生过，
# request-a 已经存在于 Worker”
#
# 然后测试我们当前关心的：
#
# continuing RUNNING request
# 的 reclaim transition。
#
#
# 返回三个对象：
#
# 1. runner
#    一个最小 fake GPUModelRunner self
#
# 2. req_states
#    真实 RequestState
#
# 3. block_tables
#    真实 BlockTables
# ============================================================

def make_runner() -> tuple[
    SimpleNamespace,
    RequestState,
    BlockTables,
]:

    # --------------------------------------------------------
    # PyTorch device descriptor。
    #
    # 这里只是在说：
    # 后续 Tensor / vLLM state 放 CUDA 上。
    #
    # 这行本身不是在分配一块大的 GPU memory。
    # --------------------------------------------------------
    device = torch.device("cuda")


    # --------------------------------------------------------
    # 创建真实 RequestState。
    #
    # max_num_reqs=1 很重要：
    #
    # 我们只创建一个 Worker request row。
    #
    # 同时 StagedWriteTensor 的 descriptor buffer capacity
    # 与 row 数相关。
    #
    # 设成 1 可以最明确地暴露之前：
    #
    # 一个 request
    # 在同一个 publication window
    # 产生两条 BlockTable staged descriptors
    #
    # 导致 capacity conflict。
    # --------------------------------------------------------
    req_states = RequestState(
        max_num_reqs=1,

        # 测试不需要真实 2048 长度，
        # 256 足够覆盖 L=94 / E=62。
        max_model_len=256,

        # 当前 test 最多只构造少量 tokens。
        max_num_batched_tokens=8,

        # 明确关闭 speculative decoding。
        num_speculative_steps=0,

        # dummy vocab size。
        vocab_size=128,

        device=device,
    )


    # --------------------------------------------------------
    # 在 RequestState 中建立 request-a。
    #
    # 这是人为模拟：
    #
    # “这个 request 已经被 Worker add_requests() 建好”
    #
    # 初始状态：
    #
    # logical:
    #   num_computed_tokens = 94
    #
    # physical:
    #   effective_kv_len = 94
    #
    # 即 reclaim 前：
    #
    #   L = E = 94
    # --------------------------------------------------------
    req_states.add_request(
        req_id="request-a",

        # prompt 长度为 94。
        prompt_len=94,

        # Python:
        #
        # range(94)
        #   -> 0,1,2,...,93
        #
        # list(range(94))
        #   -> [0,1,2,...,93]
        #
        # 这里具体 token ID 不重要，
        # 只是给 RequestState 一份合法 dummy tokens。
        all_token_ids=list(range(94)),

        # 当前 logical progress。
        num_computed_tokens=94,

        # 测试中的 dummy max generation tokens。
        max_tokens=8,
    )


    # --------------------------------------------------------
    # RequestState 使用 staged-write 模型。
    #
    # add_request() 不代表所有 GPU persistent state
    # 已经立刻 publication。
    #
    # 所以这里显式：
    #
    #     apply_staged_writes()
    #
    # 把初始化 state 真正写到 GPU persistent tensors。
    # --------------------------------------------------------
    req_states.apply_staged_writes()


    # --------------------------------------------------------
    # 创建真实 Worker BlockTables。
    #
    # 当前 single KV cache group。
    # --------------------------------------------------------
    block_tables = BlockTables(

        # 每个 physical KV block 可以容纳 16 token slots。
        block_sizes=[16],

        # 只有一个 request row。
        max_num_reqs=1,

        max_num_batched_tokens=8,

        # 当前 group 的 block table 最多放 8 个 block IDs。
        #
        # 可以理解 row 大致有：
        #
        # [ ?, ?, ?, ?, ?, ?, ?, ? ]
        max_num_blocks_per_group=[8],

        device=device,

        # kernel 使用的 block size。
        kernel_block_sizes=[16],
    )


    # --------------------------------------------------------
    # 人为初始化 Worker block row：
    #
    # [10, 11, 12, 13, 14, 15]
    #
    # 可以把它们理解成：
    #
    # B0 = block ID 10
    # B1 = block ID 11
    # B2 = block ID 12
    # B3 = block ID 13
    # B4 = block ID 14
    # B5 = block ID 15
    #
    #
    # 为什么 new_block_ids 是：
    #
    # ([10, ...],)
    #
    # 而不是：
    #
    # [10, ...]
    #
    # 因为 vLLM API 支持：
    #
    # (
    #   group0_ids,
    #   group1_ids,
    #   ...
    # )
    #
    # 当前只有一个 KV cache group，
    # 所以是长度为 1 的 tuple。
    #
    #
    # Python 语法：
    #
    # ([1,2,3],)
    #
    # 最后这个逗号非常重要，
    # 表示它是 singleton tuple。
    # --------------------------------------------------------
    block_tables.append_block_ids(
        req_index=0,
        new_block_ids=(
            [10, 11, 12, 13, 14, 15],
        ),
        overwrite=True,
    )

    # append_block_ids() 只 stage 了 GPU row write。
    # 这里才真正 publication。
    block_tables.apply_staged_writes()


    # --------------------------------------------------------
    # 等待 CUDA 上之前提交的工作全部完成。
    #
    # GPU kernel 通常是 asynchronous launch：
    #
    # CPU:
    #   launch kernel
    #   然后可以继续执行 Python
    #
    # GPU:
    #   kernel 可能仍然在运行
    #
    # 测试马上要读取 GPU state，
    # 因此需要 synchronize。
    #
    #
    # 在当前 CUDA 场景中可以近似理解为：
    #
    # torch.accelerator.synchronize()
    #
    # ≈
    #
    # torch.cuda.synchronize()
    #
    # 新 API 更 accelerator-generic。
    # --------------------------------------------------------
    torch.accelerator.synchronize()


    # --------------------------------------------------------
    # SimpleNamespace：
    #
    # 构造一个只有：
    #
    # runner.req_states
    # runner.block_tables
    #
    # 的轻量对象。
    #
    # 因为 production helper 只需要这两个。
    # --------------------------------------------------------
    return (
        SimpleNamespace(
            req_states=req_states,
            block_tables=block_tables,
        ),
        req_states,
        block_tables,
    )


# ============================================================
# 把已经 staged 的 reclaim transition 真正 publication。
#
# _commit_reclaim_transition()
# 本身主要负责：
#
#   stage BlockTables mutation
#   stage effective_kv_len mutation
#
# 这里再：
#
#   apply
#   synchronize
#
# 模拟正式 Worker metadata commit。
# ============================================================

def apply_transition(runner: SimpleNamespace) -> None:

    # Publish Worker BlockTables staged writes。
    runner.block_tables.apply_staged_writes()

    # Publish RequestState staged writes，
    # 例如 effective_kv_len:
    #
    #   94 -> 62
    runner.req_states.apply_staged_writes()

    # 等 GPU 真正完成，
    # 再进行后面的 assert。
    torch.accelerator.synchronize()


# ============================================================
# 一个公共 assertion helper。
#
# 用于检查：
#
# “测试开始时的 canonical old state 是否仍然完整”
#
# 也用于 invalid-transition test：
#
# 如果 helper 抛错，
# 检查它是否真的 fail-before-mutation。
# ============================================================

def assert_initial_state(
    req_states: RequestState,
    block_tables: BlockTables,
) -> None:

    # --------------------------------------------------------
    # CPU active frontier：
    #
    # 当前 row 有 6 个 active blocks。
    # --------------------------------------------------------
    assert block_tables.num_blocks.np[0, 0] == 6


    # --------------------------------------------------------
    # 检查 GPU BlockTable row：
    #
    # gpu[0, :6]
    #
    # Python/PyTorch slicing：
    #
    # row 0
    # columns [0,6)
    #
    # 即：
    # 0,1,2,3,4,5
    # --------------------------------------------------------
    assert torch.equal(

        block_tables.block_tables[0].gpu[0, :6],

        # 创建 expected GPU Tensor。
        torch.tensor(
            [10, 11, 12, 13, 14, 15],

            # block IDs 是 int32。
            dtype=torch.int32,

            # expected 也放 GPU 上。
            device="cuda",
        ),
    )

    # --------------------------------------------------------
    # .item()
    #
    # 把一个 scalar Tensor：
    #
    # tensor(94, device='cuda:0')
    #
    # 转成 Python int:
    #
    # 94
    #
    # 注意 production 中 GPU .item() 可能引入同步，
    # 但测试中完全可以接受。
    # --------------------------------------------------------
    assert req_states.effective_kv_len.gpu[0].item() == 94

    assert req_states.num_computed_tokens.gpu[0].item() == 94


# ============================================================
# Test 1
#
# 最基本的 canonical reclaim：
#
# Before:
#
# L = 94
# E = 94
#
# blocks:
# [10,11,12,13,14,15]
#
# reclaim:
# remove 12,13
#
# After:
#
# blocks:
# [10,11,14,15]
#
# E:
# 94 -> 62
#
# L:
# 94 -> 94
#
# 当前 step 没有额外 B6。
# ============================================================

def test_canonical_reclaim_transition() -> None:

    runner, req_states, block_tables = make_runner()

    # --------------------------------------------------------
    # 直接调用 production method。
    #
    # Python 中：
    #
    # Class.method(obj, ...)
    #
    # 可以手工把 obj 当成 self。
    #
    # 这里 runner 虽然不是完整 GPUModelRunner，
    # 但拥有：
    #
    # runner.req_states
    # runner.block_tables
    #
    # 正好满足 helper 需要。
    # --------------------------------------------------------
    GPUModelRunner._commit_reclaim_transition(
        runner,

        # request
        "request-a",

        # retained old blocks
        [10, 11, 14, 15],

        # new physical valid KV length
        62,

        # producer 认为 old active blocks = 6
        6,
    )


    # helper 只是 stage，
    # 这里真正 publication。
    apply_transition(runner)


    # --------------------------------------------------------
    # 检查 final active block count。
    #
    # old 6
    # reclaim 2
    # -> 4
    # --------------------------------------------------------
    assert block_tables.num_blocks.np[0, 0] == 4


    # GPU active prefix 应该变成：
    #
    # [10,11,14,15]
    assert torch.equal(
        block_tables.block_tables[0].gpu[0, :4],
        torch.tensor(
            [10, 11, 14, 15],
            dtype=torch.int32,
            device="cuda",
        ),
    )


    # physical:
    # 94 -> 62
    assert req_states.effective_kv_len.gpu[0].item() == 62

    # logical:
    # reclaim 不应该修改 logical progress
    assert req_states.num_computed_tokens.gpu[0].item() == 94


# ============================================================
# Test 2
#
# 这是 R1 最核心的 test。
#
# 目的：
#
# 证明：
#
# retained
# +
# same-step new block
#
# 会先在 CPU compose 成一个 final row，
#
# 然后只产生：
#
# ONE BlockTable staged descriptor。
#
#
# Before:
#
# [10,11,12,13,14,15]
# L=94
# E=94
#
# reclaim retained:
#
# [10,11,14,15]
#
# same-step Scheduler allocation:
#
# [16]
#
# final:
#
# [10,11,14,15,16]
#
# E 仍然 = 62
# ============================================================

def test_single_write_final_row_avoids_descriptor_conflict() -> None:

    runner, req_states, block_tables = make_runner()

    GPUModelRunner._commit_reclaim_transition(
        runner,
        "request-a",

        # reclaim retained
        [10, 11, 14, 15],

        # physical valid length
        62,

        # old count
        6,

        # current-step allocation delta
        #
        # single KV group:
        # ([16],)
        same_step_new_block_ids=([16],),
    )


    # --------------------------------------------------------
    # 很重要：
    #
    # 这里还没 apply GPU write，
    #
    # 但 append_block_ids()
    # 已经同步更新 CPU:
    #
    # num_blocks.np
    #
    # final row 长度：
    #
    # retained 4 + new 1 = 5
    # --------------------------------------------------------
    assert block_tables.num_blocks.np[0, 0] == 5


    # --------------------------------------------------------
    # R1 最核心 white-box assertion。
    #
    # _staged_write_indices 是 StagedWriteTensor
    # 内部记录 staged descriptors 的 Python list。
    #
    # 之前原 01A：
    #
    # reclaim overwrite
    # +
    # normal append
    #
    # 会得到：
    #
    # [0, 0]
    #
    # 即 2 descriptors。
    #
    # R1 必须是：
    #
    # [0]
    #
    # 即 1 descriptor。
    #
    #
    # 这里访问 private field 是 deliberate white-box test。
    # production code 不应该这样依赖内部实现。
    # --------------------------------------------------------
    assert (
        len(
            block_tables.block_tables[0]._staged_write_indices
        )
        == 1
    )


    # 真正 publication。
    apply_transition(runner)


    # publication 后 active count 仍然是 5。
    assert block_tables.num_blocks.np[0, 0] == 5


    # final GPU row：
    #
    # [10,11,14,15,16]
    assert torch.equal(
        block_tables.block_tables[0].gpu[0, :5],
        torch.tensor(
            [10, 11, 14, 15, 16],
            dtype=torch.int32,
            device="cuda",
        ),
    )


    # 注意：
    #
    # 这里 final row 有 5 blocks，
    # 但 E 仍然是 62。
    #
    # 不能写：
    #
    #   5 * 16 = 80
    #
    # 因为 block 16 只是 current-step capacity。
    assert req_states.effective_kv_len.gpu[0].item() == 62

    # logical 不回退。
    assert req_states.num_computed_tokens.gpu[0].item() == 94


# ============================================================
# Test 3
#
# strict no-op。
#
# 如果：
#
# retained = 原完整 row
# new_E = old_E
# same-step new = None
#
# 那什么都没有变化。
#
# 此时不应该制造任何 staged descriptor。
# ============================================================

def test_noop_identity_does_not_stage_mutation() -> None:

    runner, req_states, block_tables = make_runner()

    GPUModelRunner._commit_reclaim_transition(
        runner,
        "request-a",

        # retain all
        [10, 11, 12, 13, 14, 15],

        # E 不变
        94,

        # old count 正确
        6,
    )


    # CPU frontier 不变。
    assert block_tables.num_blocks.np[0, 0] == 6


    # --------------------------------------------------------
    # Python：
    #
    # empty list:
    #   []
    #
    # bool([]) == False
    #
    # 所以：
    #
    # assert not some_list
    #
    # 表示：
    #
    # “这个 list 必须为空”
    # --------------------------------------------------------
    assert not block_tables.block_tables[0]._staged_write_indices


    # effective_kv_len 也不应该产生 staged mutation。
    assert not req_states.effective_kv_len._staged_write_indices


    # 整个 initial state 必须没变化。
    assert_initial_state(req_states, block_tables)


# ============================================================
# Test 4
#
# 没有 reclaim，
# 但是 current step 有一个 new block。
#
# 这个 test 证明：
#
# primitive 本身可以表达：
#
# old row + new delta
#
# 通过一次 final-row overwrite 完成。
#
# 注意：
#
# future 01B 中，
# 普通 non-reclaim request
# 不应该全部走这个 helper。
#
# 正常 request 还是应该继续走：
#
# overwrite=False append
#
# 这个 test 只是在验证 primitive capability。
# ============================================================

def test_no_reclaim_with_new_delta_composes_one_final_row() -> None:

    runner, req_states, block_tables = make_runner()

    GPUModelRunner._commit_reclaim_transition(
        runner,
        "request-a",

        # retained 就是完整 old row
        [10, 11, 12, 13, 14, 15],

        # physical E 没有 shrink
        94,

        # old count
        6,

        # current step 新增 block16
        same_step_new_block_ids=([16],),
    )


    # final:
    #
    # 6 old + 1 new = 7
    assert block_tables.num_blocks.np[0, 0] == 7


    # 仍然只有 ONE descriptor。
    assert (
        len(
            block_tables.block_tables[0]._staged_write_indices
        )
        == 1
    )


    # 因为：
    #
    # old_E == new_E == 94
    #
    # 所以不应该 stage effective_kv_len mutation。
    assert not req_states.effective_kv_len._staged_write_indices


    apply_transition(runner)


    # 最终 GPU row：
    #
    # [10,11,12,13,14,15,16]
    assert torch.equal(
        block_tables.block_tables[0].gpu[0, :7],
        torch.tensor(
            [10, 11, 12, 13, 14, 15, 16],
            dtype=torch.int32,
            device="cuda",
        ),
    )


    # physical progress 没变化。
    assert req_states.effective_kv_len.gpu[0].item() == 94


# ============================================================
# Pytest 参数化测试。
#
# 相当于：
#
# 用下面 4 组输入，
# 自动重复执行：
#
# test_invalid_transition_fails_before_mutation(...)
#
#
# 每组参数：
#
# retained
# new_len
# expected_blocks
# expected error message
# ============================================================

@pytest.mark.parametrize(
    ("retained", "new_len", "expected_blocks", "message"),
    [
        # ----------------------------------------------------
        # Case 1:
        #
        # producer 说 old block count 应该是5，
        # 但 Worker 实际是6。
        #
        # stale decision fence 应该拒绝。
        # ----------------------------------------------------
        (
            [10, 11, 14, 15],
            62,
            5,
            "expected 5 blocks",
        ),

        # ----------------------------------------------------
        # Case 2:
        #
        # retained 内出现 duplicate 14。
        # ----------------------------------------------------
        (
            [10, 11, 12, 13, 14, 14],
            62,
            6,
            "unique",
        ),

        # ----------------------------------------------------
        # Case 3:
        #
        # old E=94
        # new E=63
        #
        # physical shrink:
        # 31
        #
        # 但删 2 blocks 应该 shrink：
        # 32
        #
        # 不符合 whole-block reclaim。
        # ----------------------------------------------------
        (
            [10, 11, 14, 15],
            63,
            6,
            "whole blocks",
        ),

        # ----------------------------------------------------
        # Case 4:
        #
        # new E=61
        #
        # shrink=33
        #
        # 同样不符合 2*16。
        # ----------------------------------------------------
        (
            [10, 11, 14, 15],
            61,
            6,
            "whole blocks",
        ),
    ],
)

def test_invalid_transition_fails_before_mutation(
    retained: list[int],
    new_len: int,
    expected_blocks: int,
    message: str,
) -> None:

    runner, req_states, block_tables = make_runner()


    # --------------------------------------------------------
    # pytest.raises：
    #
    # 断言 with block 内：
    #
    # 必须抛 ValueError
    #
    # 并且 error message 要 regex match:
    #
    # message
    #
    # 如果不抛异常 / 抛错类型不对 / message 不匹配，
    # test 都失败。
    # --------------------------------------------------------
    with pytest.raises(
        ValueError,
        match=message,
    ):
        GPUModelRunner._commit_reclaim_transition(
            runner,
            "request-a",
            retained,
            new_len,
            expected_blocks,
        )


    # --------------------------------------------------------
    # 关键：
    #
    # “抛异常”还不够。
    #
    # 我们还要证明：
    #
    # exception 发生之前没有 partial mutation。
    #
    # 即：
    #
    # fail-before-mutation
    # --------------------------------------------------------
    assert_initial_state(req_states, block_tables)


    # BlockTable 没有留下 staged descriptor。
    assert not block_tables.block_tables[0]._staged_write_indices


    # effective state 也没留下 staged mutation。
    assert not req_states.effective_kv_len._staged_write_indices


# ============================================================
# Test：
#
# 专门破坏：
#
# L % S == E % S
#
# invariant。
#
#
# 正常：
#
# L=94
# E=62
#
# 94-62=32
# 32%16=0
#
#
# 现在人为把 CPU logical mirror 改成：
#
# L=95
#
# 则：
#
# 95-62=33
#
# 不是 16 的整数倍。
# ============================================================

def test_logical_physical_alignment_violation_fails_before_mutation() -> None:

    runner, req_states, block_tables = make_runner()


    # --------------------------------------------------------
    # 这里只改 CPU logical mirror。
    #
    # GPU persistent logical 仍然是94。
    #
    # 这是 deliberate invalid state injection，
    # 用于精确测试 validation branch。
    # --------------------------------------------------------
    req_states.num_computed_tokens_np[0] = 95


    with pytest.raises(
        ValueError,
        match="logical/physical",
    ):
        GPUModelRunner._commit_reclaim_transition(
            runner,
            "request-a",
            [10, 11, 14, 15],
            62,
            6,
        )


    # --------------------------------------------------------
    # 注意：
    #
    # assert_initial_state()
    # 检查的是 GPU num_computed_tokens：
    #
    # 仍然是94。
    #
    # 所以能通过。
    #
    # CPU mirror 被我们人为改成95，
    # 只是为了触发 helper validation。
    # --------------------------------------------------------
    assert_initial_state(req_states, block_tables)


    assert not block_tables.block_tables[0]._staged_write_indices
    assert not req_states.effective_kv_len._staged_write_indices


# ============================================================
# Test：
#
# 两个最基本的 invalid inputs：
#
# 1. request 根本不存在
# 2. retained row 为空
# ============================================================

def test_invalid_request_and_empty_retained_ids_fail_before_mutation() -> None:

    runner, req_states, block_tables = make_runner()


    # request_id 不存在。
    with pytest.raises(
        ValueError,
        match="Unknown worker request",
    ):
        GPUModelRunner._commit_reclaim_transition(
            runner,
            "missing",
            [10],
            16,
            1,
        )


    # retained 为空。
    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        GPUModelRunner._commit_reclaim_transition(
            runner,
            "request-a",
            [],
            0,
            6,
        )


    # 原状态不能被破坏。
    assert_initial_state(req_states, block_tables)


# ============================================================
# Test：
#
# 验证 reclaim 后，
# S1 normal progress lifecycle 是否还能继续正常运行。
#
#
# 第一步 reclaim：
#
# L=94
# E=94
#
#        ↓
#
# L=94
# E=62
#
#
# 第二步模拟正常 forward 成功处理一个 token：
#
# post_update()
#
#        ↓
#
# L=95
# E=63
#
#
# 这个 test 证明：
#
# reclaim 不是只让“某一瞬间 state 看起来正确”，
# 而是能继续接回正常执行链。
# ============================================================

def test_normal_growth_after_reclaim_keeps_two_progress_states_independent() -> None:

    runner, req_states, block_tables = make_runner()


    # 先做 canonical reclaim。
    GPUModelRunner._commit_reclaim_transition(
        runner,
        "request-a",
        [10, 11, 14, 15],
        62,
        6,
    )

    apply_transition(runner)


    # --------------------------------------------------------
    # 下面直接调用 production post_update()。
    #
    # 模拟：
    #
    # 当前 batch 中 request-a
    # 正常执行并成功接受 1 个 token。
    # --------------------------------------------------------
    post_update(

        # ----------------------------------------------------
        # idx_mapping:
        #
        # 当前 batch request index
        # → RequestState persistent row index
        #
        # 当前只有 request-a，
        # 它就在 state row 0。
        #
        # dtype=int32：
        # 32-bit integer tensor。
        # ----------------------------------------------------
        idx_mapping=torch.tensor(
            [0],
            dtype=torch.int32,
            device="cuda",
        ),


        # logical persistent state tensor。
        num_computed_tokens=req_states.num_computed_tokens.gpu,


        # physical persistent state tensor。
        effective_kv_len=req_states.effective_kv_len.gpu,


        # 正常 sampler lifecycle 使用的 state。
        last_sampled_tokens=req_states.last_sampled_tokens,


        # 当前 test 不需要 penalty bin counts。
        output_bin_counts=None,


        # ----------------------------------------------------
        # sampled_tokens：
        #
        # 模拟 sampler 产生 token ID=7。
        #
        # shape 类似：
        #
        # [num_requests, sampled_per_request]
        #
        # 所以一个 request 一个 token：
        #
        # [[7]]
        # ----------------------------------------------------
        sampled_tokens=torch.tensor(
            [[7]],
            dtype=torch.int64,
            device="cuda",
        ),


        # 当前 request 实际 sampled 1 token。
        num_sampled=torch.tensor(
            [1],
            dtype=torch.int32,
            device="cuda",
        ),


        # 没有 speculative rejection。
        num_rejected=torch.tensor(
            [0],
            dtype=torch.int32,
            device="cuda",
        ),


        # ----------------------------------------------------
        # query_start_loc：
        #
        # ragged batch 常见的 prefix-sum / CSR 风格表达。
        #
        # 当前只有一个 request，
        # query length = 1。
        #
        # 所以：
        #
        # [0, 1]
        #
        # 表示：
        #
        # request 0 的 token range：
        #
        # [0,1)
        #
        #
        # 如果：
        #
        # req0 q=3
        # req1 q=2
        #
        # 则可能：
        #
        # [0,3,5]
        #
        # req0 = [0,3)
        # req1 = [3,5)
        # ----------------------------------------------------
        query_start_loc=torch.tensor(
            [0, 1],
            dtype=torch.int32,
            device="cuda",
        ),


        # persistent token ID storage。
        all_token_ids=req_states.all_token_ids.gpu,


        # persistent total length state。
        total_len=req_states.total_len.gpu,
    )


    # 等 post_update kernel 完成。
    torch.accelerator.synchronize()


    # BlockTables 本身没有变化。
    #
    # 仍然是 reclaim 后 4 blocks。
    assert block_tables.num_blocks.np[0, 0] == 4


    # logical:
    #
    # 94 + 1 = 95
    assert req_states.num_computed_tokens.gpu[0].item() == 95


    # physical:
    #
    # 62 + 1 = 63
    assert req_states.effective_kv_len.gpu[0].item() == 63


# ============================================================
# Test：
#
# 这是 R1 + S2 的连接性测试。
#
# 不只看：
#
# final row 是不是：
#
# [10,11,14,15,16]
#
# 而是进一步调用真正：
#
# BlockTables.compute_slot_mappings()
#
# 问：
#
# physical cache position 64
# 最终是不是会真正命中新 block 16？
#
#
# final Worker row：
#
# column:
# 0   1   2   3   4
#
# block ID:
# 10  11  14  15  16
#
# block_size=16
#
#
# physical pos 62：
#
# col = 62 // 16 = 3
# offset = 62 % 16 = 14
#
# col3 -> block15
#
# slot = 15*16 + 14 = 254
#
#
# physical pos 63：
#
# col3
# offset15
#
# slot=255
#
#
# physical pos64：
#
# col4
# offset0
#
# col4 -> block16
#
# slot=16*16=256
#
#
# 所以期待：
#
# [254,255,256]
# ============================================================

def test_boundary_crossing_maps_position_64_to_new_block() -> None:

    # --------------------------------------------------------
    # Python tuple unpacking：
    #
    # make_runner() 返回：
    #
    # runner
    # req_states
    # block_tables
    #
    # 这个 test 不需要 req_states。
    #
    # 所以用：
    #
    # _
    #
    # 表示“这个返回值我故意不用”。
    # --------------------------------------------------------
    runner, _, block_tables = make_runner()


    # reclaim + same-step B6。
    GPUModelRunner._commit_reclaim_transition(
        runner,
        "request-a",
        [10, 11, 14, 15],
        62,
        6,
        same_step_new_block_ids=([16],),
    )


    # publication final row。
    apply_transition(runner)


    # --------------------------------------------------------
    # 真正调用 Worker BlockTables slot mapping。
    #
    # 注意参数名仍叫 positions，
    #
    # 但在我们 P1 S2 production path 中，
    # 传入这个 API 的语义已经是：
    #
    # cache_positions
    #
    # 即 physical positions。
    #
    # 这里明确传：
    #
    # [62,63,64]
    #
    # 而不是 logical：
    #
    # [94,95,96]
    # --------------------------------------------------------
    slots = block_tables.compute_slot_mappings(

        # batch request0
        # -> block-table row0
        idx_mapping=torch.tensor(
            [0],
            dtype=torch.int32,
            device="cuda",
        ),


        # 一个 request，
        # 一共有 3 query tokens。
        #
        # range:
        # [0,3)
        query_start_loc=torch.tensor(
            [0, 3],
            dtype=torch.int32,
            device="cuda",
        ),


        # physical KV append positions。
        positions=torch.tensor(
            [62, 63, 64],
            dtype=torch.int64,
            device="cuda",
        ),


        # 当前实际 3 tokens，
        # 也没有额外 padding。
        num_tokens_padded=3,
    )


    # slot mapping kernel 是 GPU work，
    # 等它真正执行完。
    torch.accelerator.synchronize()


    # --------------------------------------------------------
    # slots 大致 shape：
    #
    # [num_kv_groups, num_tokens]
    #
    # 当前：
    #
    # group0
    # first 3 tokens
    #
    # 所以：
    #
    # slots[0, :3]
    #
    #
    # .cpu()
    #
    # CUDA Tensor -> CPU Tensor
    #
    #
    # .tolist()
    #
    # Tensor:
    #
    # tensor([254,255,256])
    #
    # ->
    #
    # Python list:
    #
    # [254,255,256]
    #
    # 这样 pytest/assert 最容易比较。
    # --------------------------------------------------------
    assert (
        slots[0, :3]
        .cpu()
        .tolist()
        == [254, 255, 256]
    )


def test_update_requests_routes_reclaim_and_publishes_physical_state() -> None:
    runner, req_states, block_tables = make_runner()
    runner._commit_reclaim_transition = MethodType(
        GPUModelRunner._commit_reclaim_transition, runner
    )
    cached = CachedRequestData(
        req_ids=["request-a"],
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[([16],)],
        num_computed_tokens=[94],
        num_output_tokens=[0],
        reclaim_transitions=[ReclaimTransitionData([10, 11, 14, 15], 62, 6)],
    )
    output = SimpleNamespace(
        scheduled_cached_reqs=cached,
        new_block_ids_to_zero=None,
        kv_cache_block_copies=None,
    )

    GPUModelRunner.update_requests(runner, output)
    assert len(block_tables.block_tables[0]._staged_write_indices) == 1
    assert block_tables.block_tables[0]._staged_write_starts == [0]

    req_states.effective_kv_len.apply_write()
    block_tables.apply_staged_writes()
    torch.accelerator.synchronize()
    assert req_states.effective_kv_len.gpu[0].item() == 62
    assert req_states.num_computed_tokens.gpu[0].item() == 94
    assert block_tables.block_tables[0].gpu[0, :5].cpu().tolist() == [
        10,
        11,
        14,
        15,
        16,
    ]


def test_update_requests_normal_path_remains_incremental() -> None:
    runner, req_states, block_tables = make_runner()
    cached = CachedRequestData(
        req_ids=["request-a"],
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[([16],)],
        num_computed_tokens=[94],
        num_output_tokens=[0],
        reclaim_transitions=[None],
    )
    output = SimpleNamespace(
        scheduled_cached_reqs=cached,
        new_block_ids_to_zero=None,
        kv_cache_block_copies=None,
    )

    GPUModelRunner.update_requests(runner, output)
    assert block_tables.block_tables[0]._staged_write_starts == [6]
    assert not req_states.effective_kv_len._staged_write_indices
    block_tables.apply_staged_writes()
    torch.accelerator.synchronize()
    assert block_tables.block_tables[0].gpu[0, :7].cpu().tolist() == [
        10,
        11,
        12,
        13,
        14,
        15,
        16,
    ]
