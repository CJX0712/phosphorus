"""树状草稿（EAGLE-2 式）的位置编码与注意力掩码构建（A-13）。

投机解码可把 K 个草稿 token 组织成树而非链：每个节点对应一个候选 token，节点的
注意力只能看见其祖先（含自身）。树宽按累积概率 `joint_prob = Π p` 分配预算，丢弃
`joint_prob < τ` 的路径（τ 默认 0.02）。

本模块只负责「给定草稿树结构，产出 position_ids 与 (n,n) 因果注意力掩码」，
不负责树的生成（由 drafter 产出候选后按概率剪枝）。验证（verify.py）在链模式下已
严格等价；树模式是对链模式的扩展，本文件提供结构正确性保证，供 test_tree_mask 验证。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TreeNode:
    token: int
    position_id: int
    parent: int = -1  # -1 表示根
    joint_prob: float = 1.0


@dataclass
class TreeSpec:
    nodes: list[TreeNode] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.nodes)

    def attention_mask(self) -> list[list[bool]]:
        """返回 (n, n) 因果掩码：mask[i][j] = True 表示节点 i 是节点 j 的祖先或自身。

        满足：对角线全 True；若 i 是 j 的祖先则 j 也是 i 的后代（下三角结构）。
        """
        n = len(self.nodes)
        # 先建祖先集合：对每个节点向上回溯到根
        ancestors: list[set[int]] = [set() for _ in range(n)]
        for j in range(n):
            p = self.nodes[j].parent
            while p != -1:
                ancestors[j].add(p)
                p = self.nodes[p].parent
        mask = [[False] * n for _ in range(n)]
        for j in range(n):
            mask[j][j] = True
            for a in ancestors[j]:
                mask[a][j] = True
        return mask

    def position_ids(self) -> list[int]:
        return [nd.position_id for nd in self.nodes]


def build_balanced_tree(
    root_token: int, depth: int, width: int, base_position: int = 0
) -> TreeSpec:
    """构建一棵「每层 width 个分支、深度 depth」的演示树（position 沿深度递增）。

    用于单测验证掩码结构；真实场景由 drafter 的候选 + 概率剪枝生成。
    """
    nodes: list[TreeNode] = [TreeNode(token=root_token, position_id=base_position, parent=-1)]
    for d in range(1, depth + 1):
        parents = [i for i, nd in enumerate(nodes) if nd.position_id == base_position + d - 1]
        for p in parents:
            for w in range(width):
                nodes.append(
                    TreeNode(
                        token=(p * 31 + w) % 100000,
                        position_id=base_position + d,
                        parent=p,
                        joint_prob=nodes[p].joint_prob * 0.5,
                    )
                )
    return TreeSpec(nodes=nodes)


def prune_by_prob(tree: TreeSpec, tau: float = 0.02) -> TreeSpec:
    """丢弃 joint_prob < tau 的节点（及其整棵子树）。"""
    keep = [nd.joint_prob >= tau for nd in tree.nodes]
    # 保留所有祖先（即便自身 < tau 也要保留以便连通）
    n = len(tree.nodes)
    ancestors: list[set[int]] = [set() for _ in range(n)]
    for j in range(n):
        p = tree.nodes[j].parent
        while p != -1:
            ancestors[j].add(p)
            p = tree.nodes[p].parent
    final_keep = [keep[j] or any(keep[a] for a in ancestors[j]) for j in range(n)]
    return TreeSpec(nodes=[tree.nodes[j] for j in range(n) if final_keep[j]])
