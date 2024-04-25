class Policy():
    def __init__(self):
        pass

    def startGame(self):
        """
        initialize all relevant variables (different from the init because a policy can be rerun multiple times)
        :return:
        """

    def choice(self):
        """
        Return a point to sample
        :return: point in which to sample
        """
        pass

    def getValue(self, value):
        """
        Update the variables after getting to know the value of the function at the chosen point
        :param value:
        :return: None
        """
        pass
