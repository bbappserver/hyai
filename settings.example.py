hydrus_api_url="http://localhost:45869/"
hydrus_api_key="replacewithyourAPIkey"

my_tags_service_key="6c6f63616c2074616773"

#For binary evaluation like archive-trash
#trash is >0.5
#archive is <0.5
#Distance from 0.5 is certainty
#higher threshold certainty has fewer false positives
#but will also skip more elmentsthat it's uncertain about
trash_thresh= 0.5 + 0.28 #it seems to be harder to be certain of trash so be more strict
archive_thresh = 0.5 - 0.22 

#Set to false to skip sending a particular type
mark_archive=True
mark_trash=True

#Choose what images you want to mark
#NOTE: duration and filetype should be left as is, the system cannot handle non static images and non-image files
#criteria=["system:no duration","system:filetype = image","-hyai:archive","-hyai:trash","system:inbox","system:limit =500000"]
criteria=["system:no duration","system:filetype = image","-hyai:archive","-hyai:trash"]

#try to evaluate this many images on your gpu at the same-time
#if insuffient VRAM to pipeline, reduce this number or consider using thumbnails
#rather than full sized iamges.
batch_size=512 

#send to hydrus when this many hashes are ready
api_batch_size=256 

#Supply a function that looks up the file given a hash as hex string
def hash_to_path(h):
    '''Converts the hash given by hydrus to a filesystem path'''
    prefix='/media/hydrus'

    use_thumbnail=False
    if use_thumbnail:
        return f'{prefix}/t{h[:2]}/{h}.thumbnail'

    p=f'{prefix}/f{h[:2]}/{h}.{ext}'
    #try the usual image file extensions
    #@HydrusDeveloper give us a better API for bulk lookup of paths
    
    for ext in ('jpg','png','gif','bmp'): 
        p=f'{prefix}/f{h[:2]}/{h}.{ext}'
        if os.path.exists(p):return p
