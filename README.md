# Case
Let's imagine we have a system/equipment/process which is required to be managed based on data. The system has sensors which measures system parameters. Suppose, we can adjust each measured system parameter except `t5`, which could be observed and could be changed by other parameters indirectly.

__Goal__:   
How to control `t5` using other parameters (`t1-t4, t6-t13`). Which values should be set for such parameters to achive specific value of `t5`? 

__About data__:   
Placed in `./data/raw/raw_data.csv`
* columns - 13 parameters
* rows - measurements

# Info
The main investigation file is `./notebooks/0.0._parshin_explore.ipynb`. Detailed descriptions are omitted for some steps. The notebook is mostly aimed to show approaches and logic. Some part of code could be (required t be) refactored.

# Contacts
Parshin Sergei / @ParshinSA / Sergei.A.P@yandex.com
