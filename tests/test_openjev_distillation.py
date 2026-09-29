"""Numerical gates for the actual soft-target loss (run in the GPU image)."""
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('peft')
from scripts.distill_openjev import cached_logits, datasets, mixed_loss


def test_soft_targets_move_student_toward_teacher_and_ignore_logit_offset():
    teacher = torch.tensor([[2., -1., .5], [-1., 3., 0.]])
    student = torch.zeros_like(teacher, requires_grad=True)
    labels = torch.tensor([0, 1])
    loss, _, _ = mixed_loss(student, teacher, labels, soft_weight=1.)
    shifted, _, _ = mixed_loss(student + 9., teacher - 17., labels, soft_weight=1.)
    assert torch.allclose(loss, shifted, atol=1e-6)
    loss.backward()
    updated = student.detach() - student.grad
    improved, _, _ = mixed_loss(updated, teacher, labels, soft_weight=1.)
    assert improved < loss
    assert updated.argmax(-1).tolist() == [0, 1]


def test_distillation_has_zero_gradient_when_student_matches_teacher():
    teacher = torch.tensor([[2., -1., .5], [-1., 3., 0.]])
    student = teacher.clone().requires_grad_(True)
    loss, _, _ = mixed_loss(student, teacher, torch.tensor([0, 1]), soft_weight=1.)
    loss.backward()
    assert abs(loss.item()) < 1e-6
    assert student.grad.abs().max() < 1e-6


def test_gold_labels_still_correct_teacher_mistakes():
    teacher = torch.tensor([[10., -3., -3.]])
    student = torch.zeros_like(teacher, requires_grad=True)
    loss, _, _ = mixed_loss(student, teacher, torch.tensor([1]), soft_weight=0.)
    loss.backward()
    assert student.grad[0, 1] < 0
    assert student.grad[0, 0] > 0


def test_teacher_cache_rejects_changed_training_data(tmp_path):
    import hashlib
    import json
    data = tmp_path / 'data'
    cache = tmp_path / 'cache'
    data.mkdir()
    cache.mkdir()
    source = data / 'ml_train.jsonl'
    source.write_text('{}\n')
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    (cache / 'manifest.json').write_text(json.dumps({
        'status': 'complete', 'data_sha256': {source.name: before}, 'logits_sha256': {},
    }))
    source.write_text('{"changed": true}\n')
    with pytest.raises(ValueError, match='different data'):
        cached_logits(cache, data)


def test_rotations_from_one_group_cannot_cross_train_dev(tmp_path):
    import json
    for split in ('train', 'dev'):
        (tmp_path / f'ml_{split}.jsonl').write_text(json.dumps({'group': 'same-dataset', 'views': []}) + '\n')
        (tmp_path / f'replay_{split}.jsonl').write_text('')
    with pytest.raises(ValueError, match='group overlap'):
        datasets(tmp_path)
