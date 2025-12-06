import pytest
import math
from bitfloat import BitFloat, F32, F64, F16, BF16, F8E4M3, F8E5M2, F4E2M1, BF24


class TestBitFloatBasics:
    """Test basic BitFloat initialization and conversion."""

    def test_init_default(self):
        """Test default initialization."""
        bf = BitFloat()
        assert bf.sign == 0
        assert bf.exponent == 0
        assert bf.mantissa == 0
        assert bf.exponent_bits == 8
        assert bf.mantissa_bits == 23

    def test_init_custom_bits(self):
        """Test initialization with custom bit widths."""
        bf = BitFloat(sign=0, exponent=5, mantissa=10, exponent_bits=5, mantissa_bits=10)
        assert bf.exponent_bits == 5
        assert bf.mantissa_bits == 10
        assert bf.exponent == 5
        assert bf.mantissa == 10

    def test_zero_positive(self):
        """Test positive zero."""
        bf = BitFloat(sign=0, exponent=0, mantissa=0)
        assert bf.to_float() == 0.0
        assert math.copysign(1, bf.to_float()) == 1.0

    def test_zero_negative(self):
        """Test negative zero."""
        bf = BitFloat(sign=1, exponent=0, mantissa=0)
        assert bf.to_float() == -0.0
        assert math.copysign(1, bf.to_float()) == -1.0

    def test_infinity_positive(self):
        """Test positive infinity."""
        bf = BitFloat(sign=0, exponent=255, mantissa=0)
        assert math.isinf(bf.to_float())
        assert bf.to_float() > 0

    def test_infinity_negative(self):
        """Test negative infinity."""
        bf = BitFloat(sign=1, exponent=255, mantissa=0)
        assert math.isinf(bf.to_float())
        assert bf.to_float() < 0

    def test_nan(self):
        """Test NaN."""
        bf = BitFloat(sign=0, exponent=255, mantissa=1)
        assert math.isnan(bf.to_float())


class TestFromFloat:
    """Test conversion from Python float to BitFloat."""

    def test_from_float_zero(self):
        """Test converting zero."""
        bf = BitFloat.from_float(0.0)
        assert bf.to_float() == 0.0

    def test_from_float_one(self):
        """Test converting 1.0."""
        bf = BitFloat.from_float(1.0)
        assert bf.to_float() == 1.0

    def test_from_float_negative(self):
        """Test converting negative number."""
        bf = BitFloat.from_float(-2.5)
        assert abs(bf.to_float() - (-2.5)) < 1e-6

    def test_from_float_infinity(self):
        """Test converting infinity."""
        bf = BitFloat.from_float(float('inf'))
        assert math.isinf(bf.to_float())

    def test_from_float_nan(self):
        """Test converting NaN."""
        bf = BitFloat.from_float(float('nan'))
        assert math.isnan(bf.to_float())

    def test_from_float_small_denormal(self):
        """Test converting very small numbers (denormal)."""
        bf = F32.from_float(1e-40)
        result = bf.to_float()
        assert result > 0
        assert result < 1e-38  # Should be denormal


class TestArithmetic:
    """Test arithmetic operations."""

    def test_add_simple(self):
        """Test simple addition."""
        a = BitFloat.from_float(1.0)
        b = BitFloat.from_float(2.0)
        result = a + b
        assert abs(result.to_float() - 3.0) < 1e-6

    def test_add_zero(self):
        """Test addition with zero."""
        a = BitFloat.from_float(5.0)
        b = BitFloat.from_float(0.0)
        result = a + b
        assert abs(result.to_float() - 5.0) < 1e-6

    def test_sub_simple(self):
        """Test simple subtraction."""
        a = BitFloat.from_float(5.0)
        b = BitFloat.from_float(3.0)
        result = a - b
        assert abs(result.to_float() - 2.0) < 1e-6

    def test_mul_simple(self):
        """Test simple multiplication."""
        a = BitFloat.from_float(2.0)
        b = BitFloat.from_float(3.0)
        result = a * b
        assert abs(result.to_float() - 6.0) < 1e-6

    def test_div_simple(self):
        """Test simple division."""
        a = BitFloat.from_float(6.0)
        b = BitFloat.from_float(2.0)
        result = a / b
        assert abs(result.to_float() - 3.0) < 1e-6

    def test_div_by_zero(self):
        """Test division by zero."""
        a = BitFloat.from_float(1.0)
        b = BitFloat.from_float(0.0)
        result = a / b
        assert math.isinf(result.to_float())

    def test_add_infinity(self):
        """Test addition with infinity."""
        a = BitFloat.from_float(float('inf'))
        b = BitFloat.from_float(1.0)
        result = a + b
        assert math.isinf(result.to_float())

    def test_mul_by_zero(self):
        """Test multiplication by zero."""
        a = BitFloat.from_float(5.0)
        b = BitFloat.from_float(0.0)
        result = a * b
        assert result.to_float() == 0.0


class TestStandardFormats:
    """Test standard floating-point format subclasses."""

    def test_f32(self):
        """Test F32 (float32) format."""
        f = F32.from_float(3.14)
        assert f.exponent_bits == 8
        assert f.mantissa_bits == 23
        assert abs(f.to_float() - 3.14) < 1e-6

    def test_f64(self):
        """Test F64 (float64) format."""
        f = F64.from_float(3.141592653589793)
        assert f.exponent_bits == 11
        assert f.mantissa_bits == 52
        assert abs(f.to_float() - 3.141592653589793) < 1e-15

    def test_f16(self):
        """Test F16 (float16) format."""
        f = F16.from_float(3.14)
        assert f.exponent_bits == 5
        assert f.mantissa_bits == 10
        assert abs(f.to_float() - 3.14) < 0.01

    def test_bf16(self):
        """Test BF16 (bfloat16) format."""
        f = BF16.from_float(3.14)
        assert f.exponent_bits == 8
        assert f.mantissa_bits == 7
        assert abs(f.to_float() - 3.14) < 0.01

    def test_f8e4m3(self):
        """Test F8E4M3 format."""
        f = F8E4M3.from_float(2.0)
        assert f.exponent_bits == 4
        assert f.mantissa_bits == 3
        assert abs(f.to_float() - 2.0) < 0.1

    def test_f8e5m2(self):
        """Test F8E5M2 format."""
        f = F8E5M2.from_float(2.0)
        assert f.exponent_bits == 5
        assert f.mantissa_bits == 2
        assert abs(f.to_float() - 2.0) < 0.1

    def test_f4e2m1(self):
        """Test F4E2M1 format."""
        f = F4E2M1.from_float(1.0)
        assert f.exponent_bits == 2
        assert f.mantissa_bits == 1

    def test_bf24(self):
        """Test BF24 custom format."""
        f = BF24.from_float(3.14159)
        assert f.exponent_bits == 8
        assert f.mantissa_bits == 15
        assert abs(f.to_float() - 3.14159) < 1e-4


class TestRepr:
    """Test string representation."""

    def test_repr_zero(self):
        """Test repr for zero."""
        bf = BitFloat(sign=0, exponent=0, mantissa=0)
        repr_str = repr(bf)
        assert "zero" in repr_str
        assert "0.0" in repr_str

    def test_repr_normal(self):
        """Test repr for normal number."""
        bf = BitFloat.from_float(1.0)
        repr_str = repr(bf)
        assert "normal" in repr_str

    def test_repr_inf(self):
        """Test repr for infinity."""
        bf = BitFloat(sign=0, exponent=255, mantissa=0)
        repr_str = repr(bf)
        assert "inf" in repr_str

    def test_repr_nan(self):
        """Test repr for NaN."""
        bf = BitFloat(sign=0, exponent=255, mantissa=1)
        repr_str = repr(bf)
        assert "nan" in repr_str


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_bit_clamping(self):
        """Test that bits are properly clamped to their ranges."""
        bf = BitFloat(sign=2, exponent=512, mantissa=0xFFFFFFFF, exponent_bits=8, mantissa_bits=23)
        assert bf.sign == 0  # 2 & 1 = 0
        assert bf.exponent == 0  # 512 & 0xFF = 0
        assert bf.mantissa == 0x7FFFFF  # Clamped to 23 bits

    def test_roundtrip_small_values(self):
        """Test roundtrip conversion for small values."""
        values = [0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 10.0]
        for val in values:
            bf = BitFloat.from_float(val)
            result = bf.to_float()
            assert abs(result - val) < 1e-6

    def test_roundtrip_negative_values(self):
        """Test roundtrip conversion for negative values."""
        values = [-0.1, -1.0, -2.5, -100.0]
        for val in values:
            bf = BitFloat.from_float(val)
            result = bf.to_float()
            assert abs(result - val) < 1e-6

    def test_mixed_format_arithmetic(self):
        """Test arithmetic between different bit widths uses first operand's format."""
        a = F32.from_float(2.0)
        b = F32.from_float(3.0)
        result = a + b
        assert result.exponent_bits == 8
        assert result.mantissa_bits == 23
