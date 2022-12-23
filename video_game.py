class VideoGameDataset:
    def __init__(self, app_id, name, release_date, developer, publisher, platform, genre, positive_ratings,
                 negative_ratings, price):
        self.app_id = app_id
        self.name = name
        self.release_date = release_date
        self.developer = developer
        self.publisher = publisher
        self.platform = platform
        self.genre = genre
        self.positive_ratings = positive_ratings
        self.negative_ratings = negative_ratings
        self.price = price

    def __str__(self):
        return "%s, %s, %s, %s, %s, %s, %s, %s" % (self.app_id, self.name, self.release_date,
                                                   self.genre, self.publisher, self.positive_ratings,
                                                   self.negative_ratings, self.price)
