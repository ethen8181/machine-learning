# Associate youtube video
# https://www.youtube.com/watch?v=0Keq3E2bbeE
# https://www.youtube.com/watch?v=hKuw-8Gjwjo
import unittest

def my_contains(elem, lst):
	"""Returns True if and only if the element is in the list"""
	return elem in lst

def my_first(lst):
	"""Returns the first element in the list"""
	return lst[0]

def bigger(lst1, lst2):
	"""
	Returns True if the sum of the element in list 1
	is greater than the sum of the elements in list 2
	"""
	return sum(lst1) > sum(lst2)


class Test(unittest.TestCase):
	"""
	class name starts with Test and inherits the unittest.TestCase,
	then you organize your test into different test cases by defining 
	different methods, note that each method's name has to start with `test`.
	two most common tests are assertEqual and assertTrue
	"""

	def setUp(self):
		"""
		if you find yourself using the same objects in multiple tests,
		then you can define them in the setUp method and call them
		"""
		self.lst1 = [1, 2, 3]
		self.lst2 = [-3, 4, 10]

	def test_contains_simple_true(self):
		"""assertTrue: pass test if it is true"""
		self.assertTrue(my_contains(elem = 3, lst = [1, 2, 3]))


	def test_first_number(self):
		"""assertEqual: pass test if the result matches"""
		self.assertEqual(my_first([1, 2, 3]), 1)

	def test_first_empty(self):
		"""
		assertRaises, makes sure the code that fails returns the 
		indicated error, you pass in the expected error type, the 
		function call and all the other arguments to that function call
		"""
		self.assertRaises(IndexError, my_first, [])

	def test_bigger_typical_true(self):
		self.assertFalse(bigger(self.lst1, self.lst2))


if __name__ == '__main__':
	# code that runs all the test
	unittest.main()


