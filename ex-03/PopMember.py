from smac.configspace import Configuration


class PopMember(object):

    def __init__(self,
                 config: Configuration,
                 age: int,
                 gender: int):
        self.config = config
        self.age = age
        self.gender = gender

    def increase_age(self):
        """ Function to increase age
        """
        self.age += 1

    def __eq__(self, other):
        return self.config == other.config


    def __hash__(self):
        return hash(self.config)
        
