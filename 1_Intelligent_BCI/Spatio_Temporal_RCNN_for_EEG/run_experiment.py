import model_exec
import setting as st
import model_def



md = model_def.model_def()
me = model_exec.model_exec()


# Training part
# cross_entropy, soft_max, data_node, label_node, wsc, bsc = md.RCNN10(train=True, channel_cnt=st.channel_cnt, time_cnt=st.time_cnt)
# me.train_RCNN(ce = cross_entropy, sm = soft_max, dn = data_node, ln = label_node, channel_cnt = st.channel_cnt, time_cnt = st.time_cnt)


## Test part
cross_entropy, soft_max, data_node, label_node, wsc, bsc = md.RCNN10(train = False, channel_cnt = st.channel_cnt, time_cnt = st.time_cnt)
# me.test_RCNN1(soft_max, data_node, wsc, bsc, time_cnt=st.time_cnt)
# me.test_RCNN2(soft_max, data_node, wsc, bsc, time_cnt=st.time_cnt)
# me.test_RCNN3(soft_max, data_node, wsc, bsc, time_cnt=st.time_cnt)
# me.test_RCNN4(soft_max, data_node, wsc, bsc, time_cnt=st.time_cnt)
# me.test_RCNN5(soft_max, data_node, wsc, bsc, time_cnt=st.time_cnt)
# me.test_RCNN6(soft_max, data_node, wsc, bsc, time_cnt=st.time_cnt)
# me.test_RCNN7(soft_max, data_node, wsc, bsc, time_cnt=st.time_cnt)
# me.test_RCNN8(soft_max, data_node, wsc, bsc, time_cnt=st.time_cnt)
me.test_RCNN9(soft_max, data_node, wsc, bsc, time_cnt=st.time_cnt)

