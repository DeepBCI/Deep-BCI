function batch_parameter_recovery_correl(names_,tlist)

load_save_dir=[pwd '/'];

res_name = 'stage2/';

interactions_names = {'(Intercept)','Uncertainty','GoalCondition','PrevAction','Uncertainty:GoalCondition','Uncertainty:PrevAction','GoalCondition:PrevAction'};
linears_names = {'(Intercept)','Uncertainty','GoalCondition','PrevAction'};

glm_betas_colike_w_sta2_inter = cell(1,max(tlist));
for ktov = tlist
    load([load_save_dir 'betas_colike_w_sta2_inter_' '_SIM' num2str(ktov) '_nanctrl_updated2.mat'],'betas_colike_w_sta2_inter');
    glm_betas_colike_w_sta2_inter{1,ktov}=betas_colike_w_sta2_inter;
end
% load([load_save_dir 'betas_colike_w_sta2_inter_behaviors_nanctrl_updated2.mat'],'betas_colike_w_sta2_inter');
% glm_betas_colike_w_sta2_inter{1, end + 1}=betas_colike_w_sta2_inter;
load([load_save_dir 'betas_colike_wo_act_inter_BEHAV.mat'],'betas_colike_w_sta2_inter');
glm_betas_colike_w_sta2_inter{1, end + 1}=betas_colike_w_sta2_inter;



b_names = {'Uncertainty', 'Goal'};
% names_= {'Kim2020','Lee2014','Pure MB','Pure MF'};
close all;
for b = 2 : 1 : 3
    t_set = 1;
    for t = tlist
        figure;
        astat = regstats(glm_betas_colike_w_sta2_inter{1,t}(:,b), glm_betas_colike_w_sta2_inter{1,end}(:,b), 'linear', 'beta'); % significant
        regression_line_ci(0.1, astat.beta, glm_betas_colike_w_sta2_inter{1,end}(:,b), glm_betas_colike_w_sta2_inter{1,t}(:,b))
        
        [R,P]=corrcoef(glm_betas_colike_w_sta2_inter{1,t}(:,b), glm_betas_colike_w_sta2_inter{1,end}(:,b));
        
        set(gcf,'Units','centi','Position',[0 1 5 5 ]);
        set(gca,'fontsize',15)
        set(gca,'fontsize',8)
        axis tight;
%         if b == 1
%             ylim([-0.2 0]);
%         else
%             ylim([-0.1 0.1]);
%         end
%         ylabel('Effect size of Behaviors')
%         xlabel(['Effect size of ' names{1,t_set}])
%         set(gca,'fontsize',11)
        set(gcf,'Units','centi','Position',[2 2 5.2 5.2]);
        set(gcf,'Units','centi','Position',[2 2 3.46 2.86]);
        grid on;
%         x_y = axis;
%         y_ = yticks;
%         if length(y_) > 5
%             y__ = y_(1:2:length(y_));
%         end
        xticks([-.5, 0, .5])
%         yticks(y__)
%         yticks(linspace(x_y(3),x_y(4),5))
        print(gcf,[load_save_dir 'ES_glm_betas_colike_w_sta2_inter_' b_names{b-1} '_' names_{1,t_set}  '.png'],'-dpng','-r2000');
        title(['R=' num2str(R(1,2)) '/// P=' num2str(P(1,2)) ]);
        set(gca,'fontsize',7)
        print(gcf,[load_save_dir 'ES_glm_betas_colike_w_sta2_inter_' b_names{b-1} '_' names_{1,t_set}  'RP.png'],'-dpng','-r500');
        t_set = t_set+1;
    end
end