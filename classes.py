import sqlite3
import os
import logging
import re
import pandas as pd

folder = r"C:\Users\kelvi\Desktop"
db_name = "test_db.db"
db = rf"{folder}\{db_name}"

# function to run a query and commit as must commit to save the data otherwise just temporary
def run_sql(conn,query,many=False):
    
    if many==True:
        conn.executemany(query)
        conn.commit()
    else:
        conn.execute(query)
        conn.commit()

conn = sqlite3.connect(db) # if does not exist will be created
#cursor = conn.cursor()

try:
    conn.execute("drop table if exists xyz;")
    
    # commit the data
    conn.commit()
    
    # create a table
    conn.execute("""
                 create table if not exists xyz (
                     id INTEGER PRIMARY KEY,
                     product TEXT,
                     item TEXT
                     )
                 """)
                 
    # commit the data
    conn.commit()
    
    # or function
    run_sql(conn,query="""
                 create table if not exists xyz (
                     id INTEGER PRIMARY KEY,
                     product TEXT,
                     item TEXT
                     )
                 """)
    
    # write some data
    conn.executemany(
        "insert or ignore into xyz (id, product, item) values (?, ?, ?)",
                [
                   (1,"abv","abc"),
                   (2,"abc","abc")
                   ],
                     )
    
    # commit the data
    conn.commit()
    
    # query the data and write to csv
    query_sql = "select * from xyz"
    df = pd.read_sql_query(query_sql, conn)
    df.to_csv( rf"{folder}\test_db.csv", index=False)
    
    # process the rows
    rows = conn.execute(query_sql).fetchall()

finally:
    # close the connection
    conn.close()

# function to run a query against specified db and return some data
def sql_fun(conn):
    # table has already been created
    
     # attach new databasetable to db
     conn.execute(f"ATTACH DATABASE '{db}' AS db1")
     
     # some new query
     query = """
     select
     a.product as a_product,
     b.product as b_product
     
     from xyz a
     left join db1.xyz b
     on a.id=b.id
     
     where a.product is not null
     """
     
     # get rows and detach
     
     # write as csv
     df = pd.read_sql_query(query, conn)
     df.to_csv( rf"{folder}\test_db_query.csv", index=False)
     
     rows = conn.execute(query).fetchall()
     conn.execute("DETACH DATABASE db1")
     
     # some things to check
     failures = [
         {
             "a_product": a_product,
             "b_product": b_product,
         }
         for a_product,b_product in rows
     ]

     failures.sort(
         key=lambda failure: (
             failure["a_product"] or "",
             failure["b_product"] or "",
         )
     )

     result = len(failures) == 0
     message = (
         f"{len(failures)} have products"
     )
     details = {
         "failures": failures,
         "field": "name",
         "actual": len(failures),
         "expected": 0,
     }

     return result, message, details               

# test to check result is as expected
def test_sql():
    t = sql_fun(conn=sqlite3.connect(db))
    # access
    assert t[0] is True # result
    assert "products" in t[1] # message
    assert "name" in t[2]["field"] # access details
    
# run test
try:
    test_sql()

#finally
# close the connection
finally:
    conn.close()

###############################################################################

import shapely
import shapely.wkt
from shapely import from_wkt
from shapely.ops import unary_union
from shapely.validation import explain_validity, make_valid
import pytest
import matplotlib.pyplot as plt

#input_wkt = "MULTIPOLYGON (((0 0, 10 0, 10 10, 0 10, 0 0)),((20 20, 21 20, 21 21, 20 21, 20 20), (2 2, 2 3, 3 3, 3 2, 2 2)))"
input_wkt = "POLYGON ((0 0, 0 2, 2 0, 2 2, 0 0))"
output_wkt = shapely.wkt.loads(input_wkt)
#assert not output_wkt.is_valid
#assert output_wkt.is_valid

invalud_shape = from_wkt(input_wkt)

typer = "poly"

if typer == "multi":
    fixed_geom = unary_union([make_valid(part) for part in invalud_shape.geoms])

    f_area = fixed_geom.area
    
    
    for poly in invalud_shape.geoms:
        fig, ax = plt.subplots()
        x,y = poly.exterior.xy
        ax.plot(x,y,color="#ff5722", linewidth=2)
        ax.fill(x,y,color="#ff5722", alpha=0.3)
else:
    fig, ax = plt.subplots()
    x,y = invalud_shape.exterior.xy
    ax.plot(x,y,color="#ff5722", linewidth=2)
    ax.fill(x,y,color="#ff5722", alpha=0.3)   

# or wktplot
#from wktplot import WKTPlot
#plot = WKTPlot()
#plot.add_shape(input_wkt)

###############################################################################

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import random
import time

class catdog:
    def __init__(self,fur,name,age):
        self.fur=fur
        self.name=name
        self.age=age
        
    def type(self):
        if self.fur == "straight":
            return "cat"
        elif self.fur == "other":
            return "dog"
        else:
            return "unknown"
        
    def noise(self):
        animal = self.type()
        if animal == "cat":
            return "meow"
        else:
            return "woof"
        
animal1 = catdog(fur="other",name="daniel",age=5)

print(animal1.type())
print(animal1.noise())

pet1 = catdog("straight", "Whiskers", 3)
print(f"{pet1.name} says {pet1.noise()} and is a {pet1.type()}")