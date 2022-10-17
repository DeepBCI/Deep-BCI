# from Session_CartPole_eeg import *
from Session_atari_mod_2stage_fewshot_0428 import *
# import egi.simple as egi
import egi3.simple as egi

def run_(subi, i):
    # ns = egi.Netstation()
    # #ns.connect("10.10.10.42", 55513)
    # ns.connect("192.168.0.2", 55513) # THIS
    # ns.sync()
    ns = None
    openAI_function_atari(subi, i, ns)
    # ns.EndSession()
    # ns.disconnect()


if __name__=="__main__":
    subi = 26 #1
    # scheduling

    # sess_list, sess_goal_list = openAI_function_atari(subi, 0)  # TODO: 다시 돌려놓기 or 한 번 돌렸으면 주석처리해놓기(오래걸림)


    # first session
    run_(subi,2) # SESS_NUM = 1, 2, 3
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

# ns = egi.Netstation()
# #ns.connect("10.10.10.42", 55513)
# ns.connect("192.168.0.2", 55513) # THIS
# ns.sync()