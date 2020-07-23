## Data preprocessing: Renaming folders and files

In order to avoid problems with non-printable characters in names of directories and filenames,
the names of the pollen dataset should be updated. The depiction of the non-printable character varies from system to system. 
On Ubuntu 18.04 it is shown as the DEL character, which is '\$177' in ASCII or '\x7f' in Unicode. On Ubuntu 16.04 it is shown as the same character, but it appears to be a non ASCII conform character, because it isnt matched by the regex '[:cntrl:]'.

### Update directory names

The non-printable character is denoted as '\x7f' (Unicode). This character and whitespaces are replaced.

| Old      | Updated
|------------|----------
| --      | \_1\_    |
| \x7f     | \_2\_   | 
| Betula\x7f   | Betula2| 
| Cupressaceae\x7f     | Cupressaceae1    | 
| No Image     | NoImage    | 
| Populus\x7f   | Populus1      | 
| Sporen - Varia   | Sporen-Varia | 
| Tilia\x7f    | Tilia1    | 
| Urticaceae\x7f    | Urticaceae1   | 

Some of the renamed directories will be considered as misspelled folders in the pollen\_classification/config/config.py file. Here you can update POLLEN\_DIR\_MATCHER-dictionary, if you renamed the directories in another way. 

### Update file names
Each png-pollen image contains the DEL-character, too. This may cause problems to properly read the filenames to create TFRecords. All filenames can be updated by running

- _find . -type f -name '*.png' -exec rename -v 's/[^\x00-\x7f]//g'  {} +_, if the non-printable character is a non-ASCII-character.
- _find . -type f -name '*.png' -exec rename -v 's/[[:cntrl:]]//g' {} +_
, if it is an ASCII-character.

which replaces each non-ASCII character in a name of a png-file. If you are unsure which of these two is the right one, try the first command first.

If you first want to simulate this command, add the -n flag to the rename command.

