import unittest
from attncopy.argument_features import string_features


class FeaturesTests(unittest.TestCase):
    def test_priority_and_boundaries(self):
        cases=[('',0,'empty'),('https://x.test/1',8,'URL'),('a@x.test',4,'email'),('abc_1',3,'identifier'),('two words',2,'natural_language_proxy'),('single',1,'other_string')]
        for text,n,kind in cases:self.assertEqual(string_features(text,n)['type_bin'],kind)
        self.assertEqual(string_features('abcd',1)['fragment_bin'],'(0,0.25]')
        self.assertEqual(string_features('abcd',2)['fragment_bin'],'(0.25,0.5]')
        self.assertEqual(string_features('abcd',4)['fragment_bin'],'(0.5,1]')
        self.assertEqual(string_features('abcd',5)['fragment_bin'],'>1')
        self.assertEqual(string_features('한글🙂',2)['characters'],3)


if __name__=='__main__':unittest.main()
