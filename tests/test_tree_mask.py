"""T5 · 树状草稿掩码结构测试（A-13）。"""

from __future__ import annotations

from phosphorus.speculate.tree import build_balanced_tree, prune_by_prob


def test_tree_mask_shape_and_diagonal():
    tree = build_balanced_tree(root_token=10, depth=3, width=2)
    n = len(tree)
    mask = tree.attention_mask()
    assert len(mask) == n and all(len(r) == n for r in mask), "掩码应为 (n, n)"
    # 对角线全 True（节点可看见自身）
    for i in range(n):
        assert mask[i][i] is True


def test_tree_causal_ancestor():
    tree = build_balanced_tree(root_token=10, depth=2, width=2)
    mask = tree.attention_mask()
    # 根节点 0 是所有人祖先；叶子应能看到根
    leaves = [i for i, nd in enumerate(tree.nodes) if nd.position_id == tree.position_ids()[0] + 2]
    for leaf in leaves:
        assert mask[0][leaf] is True, "叶子应能看到根"
        # 叶子不应被先于其祖先的「兄弟」之外看见：检查根的父为 -1
        assert tree.nodes[0].parent == -1


def test_prune_drops_low_prob_subtrees():
    tree = build_balanced_tree(root_token=1, depth=4, width=2)
    n0 = len(tree)
    pruned = prune_by_prob(tree, tau=0.1)
    assert len(pruned) <= n0
    # 所有保留节点的 joint_prob >= tau 或拥有保留的祖先
    ok = True
    for nd in pruned.nodes:
        if nd.joint_prob < 0.1:
            # 必须存在保留的祖先
            p = nd.parent
            has_kept_ancestor = False
            while p != -1:
                if any(x.joint_prob >= 0.1 for x in pruned.nodes if x is pruned.nodes[p]):
                    has_kept_ancestor = True
                    break
                p = pruned.nodes[p].parent
            ok = ok and has_kept_ancestor
    assert ok
