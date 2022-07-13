If you run the model, you get the following error:

___________________________________________________________________________________
Traceback (most recent call last):
  File "/users/suhyeon/GitHub/st-gcn-master/main.py", line 7, in <module>
    from torchlight import import_class
ImportError: cannot import name 'import_class' from 'torchlight' 
____________________________________________________________________________________


Please change this line:
_________________________________
from torchlight import [ ]
_________________________________
to:
_________________________________
from torchlight.io import [ ]
_________________________________


Thank you.
