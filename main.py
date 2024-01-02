import numpy as np
import random

# convolve2D by SamratSahoo
# https://gist.github.com/SamratSahoo/cef04a39a4033f7bec0299a10701eb95 
def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

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

def main_hot():
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
    main_hot()
