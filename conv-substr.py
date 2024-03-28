import numpy as np
from numpy.lib.stride_tricks import as_strided
import random

# ChatGPT optimized version of 2D convolution by SamratSahoo
# https://gist.github.com/SamratSahoo/cef04a39a4033f7bec0299a10701eb95
# using as_strided view into array and optimized np.tensordot implementation
def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather shapes
    xKernShape, yKernShape = kernel.shape
    xImgShape, yImgShape = image.shape

    # Calculate output shape
    xOutput = (xImgShape - xKernShape + 2 * padding) // strides + 1
    yOutput = (yImgShape - yKernShape + 2 * padding) // strides + 1

    # Pad the image
    if padding != 0:
        image = np.pad(image, pad_width=padding, mode='constant', constant_values=0)

    # Use as_strided to create a view of image data with sliding window
    output_shape = (xOutput, yOutput, xKernShape, yKernShape)
    strides = (image.strides[0]*strides, image.strides[1]*strides, image.strides[0], image.strides[1])
    convolved = as_strided(image, shape=output_shape, strides=strides)

    # Perform convolution
    output = np.tensordot(convolved, kernel, axes=((2,3), (0,1)))

    return output

def char_to_bits(char: int) -> list[int]:
    return [(char >> i) & 1 for i in range(0, 8)]

def bits_to_char(bits: list[int]) -> int:
    return sum([bits[i] * 2 ** i for i in range(0, 8)])

def str_to_one_hot(input: str) -> list[list[int]]:
    return [char_to_bits(byte) for byte in bytes(input, 'UTF-8')]

def one_hot_to_str(bits_lists: list[list[int]]) -> str:
    return bytes([bits_to_char(bits) for bits in bits_lists]).decode('UTF-8')

def reverse_hot(bits_lists: list[list[int]]) -> list[list[int]]:
    return [bits[::-1] for bits in bits_lists[::-1]]

def calculate_bits(bits_lists: list[list[int]]) -> int:
    return sum([sum(bits) for bits in bits_lists])

def main():
    # Hypothesis, convolution can be used to find substring in text
    haystack_str = 'How to Reverse a String in Python'
    needle_len = random.randint(1, len(haystack_str) - 1)
    needle_index = random.randint(0, len(haystack_str) - needle_len)
    needle_str = haystack_str[needle_index:needle_index+needle_len]
    print(f'Expecting match at index {needle_index}, needle len {needle_len}')

    haystack_hot = str_to_one_hot(haystack_str)
    needle_hot = reverse_hot(str_to_one_hot(needle_str))
    expected_bit_matches = calculate_bits(needle_hot)

    results = convolve2D(np.array(haystack_hot), np.array(needle_hot), 0)

    for index, bits_matching_np in enumerate(results):
        bits_matching = int(bits_matching_np[0])
        if bits_matching == expected_bit_matches:
            print(f'Substring matched at index {index}')
            print(f'Python index {haystack_str.index(needle_str)}')

if __name__ == '__main__':
    main()
