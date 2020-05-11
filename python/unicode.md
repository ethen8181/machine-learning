# Extremely Quick Guide to Unicode 

It's important to understand that computers can't store letters, the only thing it can store and work with are bits, 1 or 0. To use bits to represent anything beyond bits, we need rules, a.k.a. encoding to convert a letter to a sequence of bits.

Unicode is yet another encoding scheme created so that there's 1 encoding standard to unify all the various encoding standards that have been defined. In this day and age, it is recommended to use unicode.

When we open a text file and the text looks completely garbled, it is most likely we specified the wrong encoding to open the file. Or in Python 3, whenever we encounter encoding or decoding errors, e.g.

- `UnicodeDecodeError` When reading a file, we need to decode our bytes into string. If the data is not in the specified encoding when attempting to read it, an error will be thrown.
- `UnicodeEncodeError` When writing to a file, we need to encode our string into byte representation. An error will be thrown when writing out data which have no representation in the target encoding.

To solve for this, what we usually need to do is to specify the correct encoding scheme while reading or writing the data.

```python
# specify the encoding when reading the data,
# the default encoding ascii only works if it's plain English text
with open(filename, encoding='utf-8') as f:

# specify the encoding when writing the data
with open(filename, 'w', encoding='utf-8') as f:
```

Recommendation for dealing with text is software should only work with unicode strings internally, decoding the input data as soon as possible and encoding the output only at the end. This is much cleaner than throwing in random encode or decode method here and there in the program.

The notes here are more quick reference purposes, for folks looking to go more in-depth, the links in the reference section should be good starting points.

# Reference

- [Blog: Pragmatic Unicode](https://nedbatchelder.com/text/unipain.html)
- [Python Documentation: Unicode HOWTO](https://docs.python.org/3.6/howto/unicode.html)
- [Blog: What Every Programmer Absolutely, Positively Needs To Know About Encodings And Character Sets To Work With Text](http://kunststube.net/encoding/)