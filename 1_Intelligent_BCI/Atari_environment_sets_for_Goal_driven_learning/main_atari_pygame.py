# from Session_CartPole_eeg import *
from Session_atari_pygame_bf_complexity import *
# import egi.simple as egi
import egi3.simple as egi

# 'Boxing-v0'
def run_(subi, i):
    # ns = egi.Netstation()
    # ns.connect('10.10.10.42', 55513)
    # ns.sync()
    ns = None
    openAI_function_atari(subi, i, ns)
    # ns.EndSession()
    # ns.disconnect()


if __name__=="__main__":
    subi = 1
    # scheduling

    sess_list = openAI_function_atari(subi, 0)

    # first session
    run_(subi,1)
    # # 2nd session
    # run_(subi,2)
    # # 3rd session
    # run_(subi,3)
    # # 4th session
    # run_(subi,4)




# prepare for the connection
# ns = egi.Netstation()
# ns.connect('10.10.10.42', 55513)
# ns.sync()
# openAI_function(1, 1, ns)
# ns.EndSession()
# ns.disconnect()



# session1
