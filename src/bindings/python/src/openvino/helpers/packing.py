# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa

import numpy as np
from typing import Union
from openvino import Type, Shape


def pack_data(array: np.ndarray, type: Type) -> np.ndarray:
    """Represent array values as a packed openvino element type and pack them into uint8 numpy array.

    For a packed type whose bit width divides 8: standard bit packing
    For u3: Transposed packing - 8 values in 3 bytes
    For u6: Transposed packing - 4 values in 3 bytes

    If the number of elements in array is odd we pad them with zero value to be able to fit the bit
    sequence into the uint8 array.

    Example: two uint8 values - [7, 8] can be represented as uint4 values and be packed into one int8
             value - [120], because [7, 8] bit representation is [0111, 1000] will be viewed
             as [01111000], which is bit representation of [120].

    :param array: numpy array with values to pack.
    :type array: numpy array
    :param type: Type to interpret the array values. Must be a packed type (fewer than 8 bits per value).
    :type type: openvino.Type
    """
    # Handle u3 and u6 with special transposed packing
    if type == Type.u3:
        return _pack_u3(array)
    elif type == Type.u6:
        return _pack_u6(array)
    
    num_bits = _packed_bits(type)

    # A signed-integer packed type needs a two's complement container.
    minimum_regular_dtype = np.int8 if _is_signed_codes(type) else np.uint8
    casted_to_regular_type = array.astype(dtype=minimum_regular_dtype, casting="unsafe")
    if not np.array_equal(casted_to_regular_type, array):
        raise RuntimeError(f'The conversion of array "{array}" to dtype' f' "{casted_to_regular_type}" results in rounding')

    data_size = casted_to_regular_type.size
    num_values_fitting_into_uint8 = 8 // num_bits
    pad = (-data_size) % num_values_fitting_into_uint8

    flattened = casted_to_regular_type.flatten()
    padded = np.concatenate((flattened, np.zeros([pad], dtype=minimum_regular_dtype)))  # type: ignore
    assert padded.size % num_values_fitting_into_uint8 == 0

    bit_order_little = (padded[:, None] & (1 << np.arange(num_bits)) > 0).astype(minimum_regular_dtype)
    bit_order_big = np.flip(bit_order_little, axis=1)  # type: ignore
    bit_order_big_flattened = bit_order_big.flatten()

    return np.packbits(bit_order_big_flattened)


def unpack_data(array: np.ndarray, type: Type, shape: Union[list, Shape]) -> np.ndarray:
    """Extract openvino element type values from array into new uint8/int8 array given shape.

    For a packed type whose bit width divides 8: standard bit unpacking
    For u3: Transposed unpacking - 8 values from 3 bytes
    For u6: Transposed unpacking - 4 values from 3 bytes

    Example: uint8 value [120] can be represented as two u4 values and be unpacked into [7, 8]
             because [120] bit representation is [01111000] will be viewed as [0111, 1000],
             which is bit representation of [7, 8].

    :param array: numpy array to unpack.
    :type array: numpy array
    :param type: Type to extract from array values. Must be a packed type (fewer than 8 bits per value).
    :type type: openvino.Type
    :param shape: the new shape for the unpacked array.
    :type shape: Union[list, openvino.Shape]
    """
    # Handle u3 and u6 with special transposed unpacking
    if type == Type.u3:
        return _unpack_u3(array, shape)
    elif type == Type.u6:
        return _unpack_u6(array, shape)
    
    num_bits = _packed_bits(type)
    unpacked = np.unpackbits(array.view(np.uint8))
    shape = list(shape)
    if num_bits == 1:
        return np.resize(unpacked, shape)
    else:
        unpacked = unpacked.reshape(-1, num_bits)
        padding_shape = (unpacked.shape[0], 8 - num_bits)
        padding = np.ndarray(padding_shape, np.uint8)  # type: np.ndarray
        if _is_signed_codes(type):
            for axis, bits in enumerate(unpacked):
                if bits[0] == 1:
                    padding[axis] = np.ones((padding_shape[1],), np.uint8)
                else:
                    padding[axis] = np.zeros((padding_shape[1],), np.uint8)
        else:
            padding = np.zeros(padding_shape, np.uint8)
        padded = np.concatenate((padding, unpacked), 1)  # type: ignore
        packed = np.packbits(padded, 1)
        if _is_signed_codes(type):
            return np.resize(packed, shape).astype(dtype=np.int8)
        else:
            return np.resize(packed, shape)


def _packed_bits(type: Type) -> int:
    """The bit width of a packed type the standard bit packing handles, whose width divides a byte."""
    num_bits = type.bitwidth
    assert num_bits and num_bits < 8 and 8 % num_bits == 0, "The standard bit packing handles a packed type whose width divides 8 bits"
    return num_bits


def _is_signed_codes(type: Type) -> bool:
    """Whether a packed type's values are two's complement, so they need a signed container and sign extension."""
    return type.integral_number and type.signed


def _pack_u3(array: np.ndarray) -> np.ndarray:
    """Pack u3 values using transposed packing scheme.

    8 values (each 3 bits) are packed into 3 bytes:
    - Byte 0: bits [1:0] of values 0-3 (4 values * 2 bits = 8 bits)
    - Byte 1: bits [1:0] of values 4-7 (4 values * 2 bits = 8 bits)
    - Byte 2: bits [2] of all 8 values (8 values * 1 bit = 8 bits)
    """
    array = array.astype(np.uint8, casting="unsafe").flatten()
    # Pad to multiple of 8
    pad = (-len(array)) % 8
    if pad:
        array = np.concatenate([array, np.zeros(pad, dtype=np.uint8)])

    groups = array.reshape(-1, 8)
    result = []

    for group in groups:
        # Extract lower 2 bits and MSB for each value
        lower_bits = group & 0x03  # bits [1:0]
        msb = (group >> 2) & 0x01  # bit [2]

        # Pack into 3 bytes
        byte0 = (lower_bits[0] << 6) | (lower_bits[1] << 4) | (lower_bits[2] << 2) | lower_bits[3]
        byte1 = (lower_bits[4] << 6) | (lower_bits[5] << 4) | (lower_bits[6] << 2) | lower_bits[7]
        byte2 = (msb[0] << 7) | (msb[1] << 6) | (msb[2] << 5) | (msb[3] << 4) | \
                (msb[4] << 3) | (msb[5] << 2) | (msb[6] << 1) | msb[7]

        result.extend([byte0, byte1, byte2])

    return np.array(result, dtype=np.uint8)


def _unpack_u3(array: np.ndarray, shape: Union[list, Shape]) -> np.ndarray:
    """Unpack u3 values using transposed unpacking scheme.
    
    3 bytes are unpacked into 8 values (each 3 bits).
    """
    array = array.view(np.uint8)
    shape = list(shape)

    # Process 3 bytes at a time
    result = []
    for i in range(0, len(array), 3):
        if i + 2 >= len(array):
            break
        byte0, byte1, byte2 = array[i:i+3]

        # Unpack lower 2 bits from first two bytes
        val0 = (byte0 >> 6) & 0x03
        val1 = (byte0 >> 4) & 0x03
        val2 = (byte0 >> 2) & 0x03
        val3 = byte0 & 0x03
        val4 = (byte1 >> 6) & 0x03
        val5 = (byte1 >> 4) & 0x03
        val6 = (byte1 >> 2) & 0x03
        val7 = byte1 & 0x03

        # Unpack MSBs from third byte
        msb0 = (byte2 >> 7) & 0x01
        msb1 = (byte2 >> 6) & 0x01
        msb2 = (byte2 >> 5) & 0x01
        msb3 = (byte2 >> 4) & 0x01
        msb4 = (byte2 >> 3) & 0x01
        msb5 = (byte2 >> 2) & 0x01
        msb6 = (byte2 >> 1) & 0x01
        msb7 = byte2 & 0x01

        # Combine to form 3-bit values
        result.extend([
            val0 | (msb0 << 2), val1 | (msb1 << 2), val2 | (msb2 << 2), val3 | (msb3 << 2),
            val4 | (msb4 << 2), val5 | (msb5 << 2), val6 | (msb6 << 2), val7 | (msb7 << 2)
        ])

    result = np.array(result, dtype=np.uint8)
    return np.resize(result, shape)


def _pack_u6(array: np.ndarray) -> np.ndarray:
    """Pack u6 values using transposed packing scheme.

    4 values (each 6 bits) are packed into 3 bytes:
    - Byte 0: bits [3:0] of values 0-1 (2 values * 4 bits = 8 bits)
    - Byte 1: bits [3:0] of values 2-3 (2 values * 4 bits = 8 bits)
    - Byte 2: bits [5:4] of all 4 values (4 values * 2 bits = 8 bits)
    """
    array = array.astype(np.uint8, casting="unsafe").flatten()
    # Pad to multiple of 4
    pad = (-len(array)) % 4
    if pad:
        array = np.concatenate([array, np.zeros(pad, dtype=np.uint8)])

    groups = array.reshape(-1, 4)
    result = []

    for group in groups:
        lower_bits = group & 0x0F  # bits [3:0]
        upper_bits = (group >> 4) & 0x03  # bits [5:4]

        # Pack into 3 bytes
        byte0 = (lower_bits[0] << 4) | lower_bits[1]
        byte1 = (lower_bits[2] << 4) | lower_bits[3]
        byte2 = (upper_bits[0] << 6) | (upper_bits[1] << 4) | (upper_bits[2] << 2) | upper_bits[3]

        result.extend([byte0, byte1, byte2])

    return np.array(result, dtype=np.uint8)


def _unpack_u6(array: np.ndarray, shape: Union[list, Shape]) -> np.ndarray:
    """Unpack u6 values using transposed unpacking scheme.
    
    3 bytes are unpacked into 4 values (each 6 bits).
    """
    array = array.view(np.uint8)
    shape = list(shape)
    num_values = int(np.prod(shape))
    
    # Process 3 bytes at a time
    result = []
    for i in range(0, len(array), 3):
        if i + 2 >= len(array):
            break
        byte0, byte1, byte2 = array[i:i+3]
        
        # Unpack lower 4 bits from first two bytes
        val0 = (byte0 >> 4) & 0x0F
        val1 = byte0 & 0x0F
        val2 = (byte1 >> 4) & 0x0F
        val3 = byte1 & 0x0F
        
        # Unpack upper 2 bits from third byte
        upper0 = (byte2 >> 6) & 0x03
        upper1 = (byte2 >> 4) & 0x03
        upper2 = (byte2 >> 2) & 0x03
        upper3 = byte2 & 0x03
        
        # Combine to form 6-bit values
        result.extend([
            val0 | (upper0 << 4), val1 | (upper1 << 4),
            val2 | (upper2 << 4), val3 | (upper3 << 4)
        ])
    
    result = np.array(result, dtype=np.uint8)
    return np.resize(result, shape)
