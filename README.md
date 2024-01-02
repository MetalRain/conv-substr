# Substring from convolution

3Blue1Brown video "But what is a convolution?" gave me idea: Can you use convolution to find substring in string?

This repo explores this idea and maybe later compares it to other solutions.

## DNA base alignment

I found out that DNA base sequences are aligned by using [Burrows-Wheeler Alignment](
https://mr-easy.github.io/2019-12-19-burrows-wheeler-alignment-part-1/) could convolution be useful here, since it already gives us quality of match.

## References

But what is a convolution? by 3Blue1Brown
https://www.youtube.com/watch?v=KuXjwB4LzSA

Easy to understand 2D convolution implementation by @SamratSahoo
https://gist.github.com/SamratSahoo/cef04a39a4033f7bec0299a10701eb95

Burrows-Wheeler Alignment by Mr. Easy
https://mr-easy.github.io/2019-12-19-burrows-wheeler-alignment-part-1/
