import pymongo
mongoclient = pymongo.MongoClient('127.0.0.1')
#连接数据库
db = mongoclient['recommend']
dbcol=db["bookinfo"]
dbcol.delete_many({"comments":{"$size":0}})