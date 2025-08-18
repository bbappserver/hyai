import hydrus_api
import hydrus_api.utils
import os,time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import settings

topless=True
checkpoint_path= 'best_model.topless.keras' if topless else 'best_model.keras'

api_key=settings.api_key
api_url=settings.api_url

mark_archive=settings.mark_archive
mark_trash=settings.mark_trash
criteria=settings.criteria
#Approximaly 75% of this number is the amount of memory the number of gibibytes(GiB) it wil take
batch_size=settings.batch_size # How many images should be preloaded for the GPU, if you set this too high bad things happen
api_batch_size=256 #Wait for this many entries before sending to hydrus, creates less traffic and less transaction overhead

hash_to_path=settings.hash_to_path

MY_TAGS_SERVICE_KEY=settings.my_tags_service_key

client = hydrus_api.Client(api_key,api_url)
REQUIRED_PERMISSIONS = {
    hydrus_api.Permission.ADD_TAGS,
    hydrus_api.Permission.SEARCH_FILES,
}

if not hydrus_api.utils.verify_permissions(client, REQUIRED_PERMISSIONS):
    print("The API key does not grant all required permissions:", REQUIRED_PERMISSIONS)
    #sys.exit(ERROR_EXIT_CODE)

#No seriously don't push these to the PTR you imbecile
assert(MY_TAGS_SERVICE_KEY != 'c8d5413efe18a972285dcf98fc55084e47511e36fb1f4033b0b8984ccb3f54c8')



if os.path.exists(checkpoint_path):
    print(f"Loading existing checkpoint from {checkpoint_path}")
    model = tf.keras.models.load_model(checkpoint_path)
#Get images exclude animated gifs, and already tagged.
#criteria=["system:no duration","system:filetype = image","-hyai:trash","-hyai:archive"]

print("Getting hashes from hydrus")
resp= client.search_files(criteria,return_hashes=True)
all_file_ids=resp['hashes']
hash_idx=0
N=len(all_file_ids)
print(f'Will process {N} hashes.')




import glob
import itertools
import os.path
from math import ceil

hash_idx=0

pending_archive=[]
pending_trash=[]



def generate_image_paths(all_file_hashes):
    for x in all_file_hashes:
        p= hash_to_path(x) #exclude paths we failed tpo find yield others
        if p: yield p
 



def load_image(path, image_size, num_channels, interpolation, crop_to_aspect_ratio=False):
    """Load an image from a path and resize it."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(
        img, channels=num_channels, expand_animations=False
    )
#     img=tf.keras.utils.load_img(path)
    if crop_to_aspect_ratio:
        img = image_utils.smart_resize(
            img, image_size, interpolation=interpolation
        )
    else:
        img = tf.image.resize(img, image_size, method=interpolation)
    img.set_shape((image_size[0], image_size[1], num_channels))
    return img

def send_tags(hashes,tag):
    global client
    print(f"send {tag}")
    hashes= [all_file_ids[x] for x in hashes]
    if not hashes: return
    client.add_tags(hashes,service_keys_to_tags= {MY_TAGS_SERVICE_KEY : [tag]})
    time.sleep(2)#hydrus can't handle too many requests and will just disconnect so throttle

def put_result(label,index,final=False):
    global pending_trash,pending_archive
    #print ((label,all_file_ids[index]))
    if label[0]=='T':
        pending_trash.append(index)
        if final or len(pending_trash)>=api_batch_size:
            send_tags(pending_trash,'hyai:trash')
            pending_trash=[]
    else:
        pending_archive.append(index)
        if final or len(pending_archive)>=api_batch_size:
            send_tags(pending_archive,'hyai:archive')
            pending_archive=[]

#Use bilinear iflteringto make a 3 channel 224x224x3 image       
args=( (224,224),3, 'bilinear')

# do the same transformations on images as in training
path_ds = tf.data.Dataset.from_generator( lambda :generate_image_paths(all_file_ids),output_types=tf.string).prefetch(buffer_size=2048)
img_ds = path_ds.map(
    lambda x: load_image(x, *args), num_parallel_calls=tf.data.AUTOTUNE
).ignore_errors()
img_ds  = img_ds.map(lambda x: (tf.keras.applications.mobilenet_v2.preprocess_input(x),))
img_ds = img_ds.batch(batch_size).prefetch(buffer_size= int(16))


low_thresh = settings.archive_thresh
high_thresh = settings.trash_thresh
j=0
for image_batch in img_ds:
    #if j>4: break
    predicted_batch = model.predict(image_batch)
    #predicted_id = np.argmax(predicted_batch, axis=-1)
    #predicted_label_batch = class_names[predicted_id]

    for n in range(len(predicted_batch)):
        y=predicted_batch[n][0]
        #TF does binary classifcation in alphabetical order so archive would be <0.5
        if mark_archive and y<low_thresh:
            put_result("Archive",hash_idx)
        elif y>high_thresh:
            put_result("Trash",hash_idx)
        hash_idx+=1 #todo it's really sloppy to just use a global index into the lsit pulled form hydrus but it works
    
    del image_batch
    time.sleep(0.2) # without this the os will become unresponsive because you alllocate too much too quickly

#finally send anything pending
send_tags(pending_trash,'hyai:trash')
send_tags(pending_archive,'hyai:archive')
