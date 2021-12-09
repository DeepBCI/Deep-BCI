# from Session_CartPole_eeg import *
from Session_atari_pygame_C_save import *
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

    # sess_list, sess_goal_list = openAI_function_atari(subi, 0)  # TODO: 다시 돌려놓기 or 한 번 돌렸으면 주석처리해놓기(오래걸림 ㅎ)


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
