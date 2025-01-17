import os

src = os.path.dirname(os.path.abspath(__file__))
codebase = os.path.abspath(os.path.join(src, os.pardir))
tests = os.path.abspath(os.path.join(codebase, 'tests'))
fixtures = os.path.abspath(os.path.join(tests, 'fixtures'))
fixtures_in = os.path.abspath(os.path.join(fixtures, 'inputs'))
fixtures_out = os.path.abspath(os.path.join(fixtures, 'expected_outputs'))
