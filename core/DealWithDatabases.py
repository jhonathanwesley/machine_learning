# %%
from sqlalchemy import create_engine


engine = create_engine('sqlite:///../data/dataset.db', echo=True)


connection = engine.connect()

print(connection.connection.dbapi_connection)

connection.close()

