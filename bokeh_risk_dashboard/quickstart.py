from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()

gauth.LoadCredentialsFile("mycreds.txt")
if gauth.credentials is None:
    # Authenticate if they're not there
    gauth.LocalWebserverAuth()
elif gauth.access_token_expired:
    # Refresh them if expired
    gauth.Refresh()
else:
    # Initialize the saved creds
    gauth.Authorize()
# Save the current credentials to a file
gauth.SaveCredentialsFile("mycreds.txt")

drive = GoogleDrive(gauth)

drivefolder='https://drive.google.com/open?id=1OvpQZdt5TYmwsIDBcILHP5Zez6PSdDWE'

#folder_metadata = {
#    'title' : 'EndowDashboard',
    # The mimetype defines this new file as a folder, so don't change this.
#    'mimeType' : 'application/vnd.google-apps.folder'
#}
#folder = drive.CreateFile(folder_metadata)
#folder.Upload()


#folder_title = folder['title']
#folder_id = folder['id']
#print('title: %s, id: %s' % (folder_title, folder_id))

#f = drive.CreateFile({"parents": [{"kind": "drive#fileLink", "id": folder_id}]})

tgt_folder_id='1OvpQZdt5TYmwsIDBcILHP5Zez6PSdDWE'

f = drive.CreateFile({'title':'dummy2.csv', 'mimeType':'text/csv',
        "parents": [{"kind": "drive#fileLink","id": tgt_folder_id}]})

f.SetContentFile('README.txt')
f.Upload()
