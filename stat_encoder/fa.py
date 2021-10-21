class FAEncoder(object):
    '''
    Factor analysis-based encoder
    '''
    def __init__(self):
        # TODO: fa encoder
        self.__data = None
        self.A = None
        pass


    def fit(self, data):
        '''
        params:
        @data: array-like object with the following structure:
            [
                [x00, x01, x02, ..., x0p-1],
                ...,
                [xn-10, xn-11, xn-12, ..., xn-1p-1]
            ]
            in which there are n samples that each has p features
        '''
        pass


    def encode(self, data):
        pass
