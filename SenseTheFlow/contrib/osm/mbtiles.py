import gzip
import sqlite3
import mapbox_vector_tile
from io import BytesIO


def uncompress(data):
    fp = BytesIO(data)
    gz = gzip.GzipFile(fileobj=fp, mode='r')
    data = gz.read()
    gz.close()
    return data
    
def connect(filename):
    return sqlite3.connect(filename)

def parse(rows):
    for data, row, column in rows:
        try: data = uncompress(data)
        except: pass
        data = mapbox_vector_tile.decode(data)
        yield data, row, column
    
def get_first(conn, zoom_level):
    cursor = conn.cursor()
    cursor.execute('SELECT tile_data, tile_row, tile_column from tiles WHERE zoom_level=? LIMIT 1', (zoom_level,))
    return parse([cursor.fetchone()])

def get_all(conn, zoom_level):
    cursor = conn.cursor()
    cursor.execute('SELECT tile_data, tile_row, tile_column FROM tiles WHERE zoom_level=?', (zoom_level,))
    while True:
        data = cursor.fetchmany(1000)
        if not data:
            break
            
        for res in parse(data):
            yield res

def get_count(conn, zoom_level):
    try:
        cursor = conn.cursor()
        cursor.execute('select MAX(_ROWID_) FROM map WHERE zoom_level=? LIMIT 1', (zoom_level,))
    except:
        cursor = conn.cursor()
        cursor.execute('select COUNT(tile_row) FROM tiles WHERE zoom_level=? LIMIT 1', (zoom_level,))
        
    return cursor.fetchone()[0]

def get_bounds(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM metadata;')
    data = cursor.fetchall()
    
    for row in data:
        if row[0] == 'bounds':
            return tuple(map(float, row[1].split(',')))
