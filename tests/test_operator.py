import unittest
import torch
from cable import CABLE2D


class TestCABLE(unittest.TestCase):
    def test_shape_gradient_and_parameters(self):
        m = CABLE2D(32, groups=4)
        x = torch.randn(2, 32, 16, 16, requires_grad=True)
        y = m(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertEqual(sum(p.numel() for p in m.parameters()), 1568)
        y.square().mean().backward()
        self.assertIsNotNone(x.grad)

    def test_zero_delta_is_identity(self):
        m = CABLE2D(32)
        with torch.no_grad():
            for p in m.parameters(): p.zero_()
        x = torch.randn(1, 32, 8, 8)
        torch.testing.assert_close(m(x), x)

    def test_invalid_grouping(self):
        with self.assertRaises(ValueError): CABLE2D(30, groups=4)


if __name__ == '__main__': unittest.main()
