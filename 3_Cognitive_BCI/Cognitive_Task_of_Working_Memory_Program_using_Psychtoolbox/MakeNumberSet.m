function [Number,Cue,Target_Output,First]=MakeNumberSet(trial,block,task)
% trial=10;  block=4*3;  task=[0 1 2] or [0 1]
% Number(:,:,1) = task (0,1,2)
% Number(:,:,2) = answers (1&2&3) - key response
% Number(block*trial*2)
% First (cblock*1)=Answers digits
% Target(:,:,1:3)(block*trial*4) = answer_1back,lure,times,answer_2back
% Target_Output(:,:,1:3)(block*trial*3) = randomized targets
% Cue (block*trial) = 1 digit
% Constraint: 
% when trial=1, Number(:,:,1)=0 & 1
%  90 < Cue < 100
% target (answer1back, lure, times,answer2back)
% block=12;     trial=10;
Target=nan(block,trial,4);  Target_Output=nan(block,trial,3);
Number=nan(block,trial,2);
Cue=randi([2,9],[block,trial]); First=randi([min(ceil(sum(Cue,2)))+210,299],[block,1]); 
times=randi([1,4],[block,trial]);
cand_time=[1,3,5,7];
times=cand_time(times);
Number(:,:,1)=repmat(Shuffle(repmat(task,1,block/length(task))),trial,1)';
Number(:,:,2)=Shuffle(repmat(repmat([1,2,3],1,block/length(task)),trial,1)');
Target(:,:,3)=Cue.*times;
Mans=zeros(1,2);
for i=1:block
    Mans(1)=First(i);
   for j=1:trial           
       switch Number(i,j,1)
           case 0             
               ans=Target(i,j,3);
               Target_Output(i,j,Number(i,j,2))=ans;
               dummy_rand=[1,2,3];
               dummy_rand(Number(i,j,2))=[];
               dummy_rand=Shuffle(dummy_rand);
               if ans<=11
               Target_Output(i,j,dummy_rand(1))=ans+1;
               Target_Output(i,j,dummy_rand(2))=ans+2;
               elseif ans>11 && ans<=23
                   Target_Output(i,j,dummy_rand(1))=ans-2;
                   Target_Output(i,j,dummy_rand(2))=ans+2;
               else
                   Target_Output(i,j,dummy_rand(1))=ans-4;
                   Target_Output(i,j,dummy_rand(2))=ans+11;
               end
               Mans(2)=Mans(1);
               Mans(1)=ans; 
               
           case 1
               ans=Mans(1)-Cue(i,j);
               Target(i,j,1)=ans;
               Target(i,j,2)=ans+randi([1 9]);
               
               Target_Output(i,j,Number(i,j,2))=ans;
               
               dummy_rand=[1,2,3];
               dummy_rand(Number(i,j,2))=[];
               dummy_rand=Shuffle(dummy_rand);
               if ans<=11
               Target_Output(i,j,dummy_rand(1))=ans+1;
               Target_Output(i,j,dummy_rand(2))=ans+3;
               elseif ans>11 && ans<=23
                   Target_Output(i,j,dummy_rand(1))=ans-3;
                   Target_Output(i,j,dummy_rand(2))=ans+3;
               else
                   Target_Output(i,j,dummy_rand(1))=ans-3;
                   Target_Output(i,j,dummy_rand(2))=ans+6;
               end
               Mans(2)=Mans(1);
               Mans(1)=ans;
           case 2
               if j==1
                  Number(i,j,1)=0;
    
               ans=First(i)-randi(9);
               
               Target_Output(i,j,Number(i,j,2))=ans;
               dummy_rand=[1,2,3];
               dummy_rand(Number(i,j,2))=[];
               dummy_rand=Shuffle(dummy_rand);
               if ans<=11
               Target_Output(i,j,dummy_rand(1))=ans;
               Target_Output(i,j,dummy_rand(2))=ans;
               elseif ans>11 && ans<=23
                   Target_Output(i,j,dummy_rand(1))=ans;
                   Target_Output(i,j,dummy_rand(2))=ans;
               else
                   Target_Output(i,j,dummy_rand(1))=ans;
                   Target_Output(i,j,dummy_rand(2))=ans;
               end
               Mans(2)=Mans(1);
               Mans(1)=ans;
               else
               ans=Mans(2)-Cue(i,j);
               
               Target(i,j,1)=ans;
               Target(i,j,2)=ans+randi([1 9]);
               
               Target_Output(i,j,Number(i,j,2))=ans;
               
               dummy_rand=[1,2,3];
               dummy_rand(Number(i,j,2))=[];
               dummy_rand=Shuffle(dummy_rand);
               if ans<=11
               Target_Output(i,j,dummy_rand(1))=ans+11;
               Target_Output(i,j,dummy_rand(2))=ans+23;
               elseif ans>11 && ans<=23
                   Target_Output(i,j,dummy_rand(1))=ans-3;
                   Target_Output(i,j,dummy_rand(2))=ans+23;
               else
                   Target_Output(i,j,dummy_rand(1))=ans-3;
                   Target_Output(i,j,dummy_rand(2))=ans+11;
               end
               Mans(2)=Mans(1);
               Mans(1)=ans;
               end 
       end
   end
end











