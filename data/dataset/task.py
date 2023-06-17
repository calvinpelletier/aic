


def create_task_data():
    db = GameDatabase(db_name, train=False)
    for game in db.game_iter():
