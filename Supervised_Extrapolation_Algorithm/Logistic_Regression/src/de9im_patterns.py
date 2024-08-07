
###########################################################

##########  Optimized #####################################

###########################################################

DIMS = {
    'F': frozenset('F'),
    'T': frozenset('012'),
    '*': frozenset('F012'),
    '0': frozenset('0'),
    '1': frozenset('1'),
    '2': frozenset('2'),
}

class Pattern:
    def __init__(self, pattern_string):
        self.pattern = tuple(pattern_string.upper())

    def __str__(self):
        return ''.join(self.pattern)

    def __repr__(self):
        return f"DE-9IM pattern: '{str(self)}'"

    def matches(self, matrix_string):
        matrix = tuple(matrix_string.upper())
        return all(m in DIMS[p] for p, m in zip(self.pattern, matrix))

class AntiPattern:
    def __init__(self, anti_pattern_string):
        self.anti_pattern = tuple(anti_pattern_string.upper())

    def __str__(self):
        return '!' + ''.join(self.anti_pattern)

    def __repr__(self):
        return f"DE-9IM anti-pattern: '{str(self)}'"

    def matches(self, matrix_string):
        matrix = tuple(matrix_string.upper())
        return not all(m in DIMS[p] for p, m in zip(self.anti_pattern, matrix))

class NOrPattern:
    def __init__(self, pattern_strings):
        self.patterns = [tuple(s.upper()) for s in pattern_strings]

    def __str__(self):
        return '||'.join([''.join(s) for s in self.patterns])

    def __repr__(self):
        return f"DE-9IM or-pattern: '{str(self)}'"

    def matches(self, matrix_string):
        matrix = tuple(matrix_string.upper())
        return any(all(m in DIMS[p] for p, m in zip(pattern, matrix)) for pattern in self.patterns)

# Familiar names for patterns or patterns grouped in logical expression
contains = Pattern('T*****FF*')
crosses_lines = Pattern('0********')
crosses_1 = Pattern('T*T******')
crosses_2 = Pattern('T*****T**')
disjoint = AntiPattern('FF*FF****')
equal = Pattern('T*F**FFF*')
intersects = AntiPattern('FF*FF****')
overlaps1 = Pattern('T*T***T**')
overlaps2 = Pattern('1*T***T**')
touches = NOrPattern(['FT*******', 'F**T*****', 'F***T****'])
within = Pattern('T*F**F***')
covered_by = NOrPattern(['T*F**F***', '*TF**F***', '**FT*F***', '**F*TF***'])
covers = NOrPattern(['T*****FF*', '*T****FF*', '***T**FF*', '****T*FF*'])
