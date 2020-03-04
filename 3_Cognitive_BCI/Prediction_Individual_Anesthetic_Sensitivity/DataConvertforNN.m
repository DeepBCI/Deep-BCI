% DataConvertforNN.m:
% 
% NN에 사용할 수 있게 상위그룹과 하위그룹으로 나눈다. 
% K-means로 나눈다.
% High_P_DataSet_All.m으로 저장
% 약재의 농도와 종류에 따라 High_P를 
% Medium_P, Low_P, High_M, Medium_M, Low_M
% 로 대체해서 저장
%
% author: Young-Seok Kweon
% created: 2019.06.17
%% initialize

clc; clear; close all;
%% value setting for data setting

basic_dir='E:\data_convert\';
basic_dir2='E:\data_result\';
Type_filename={'High_P_','Medium_P_','Low_P_';...
    'High_M_','Medium_M_','Low_M_'};

name_s={'Sensitivity'};
n=10;


%% data setting and load
sampling_frequency=1000;

for agent=1:size(Type_filename,1)
    
    for state=1:size(Type_filename,2)

        name=strcat(Type_filename{agent,state},name_s{1});
        sens=load([basic_dir2 name]);

        ce=sens.result.x(1,:);
        bis=sens.result.x(3,:);
        time=sens.result.x(2,:);

        I=clustering(ce',2); % sensitivity가 낮은 그룹이 1 (ce는 높음)
        data.y=I;

        % medium group의 첫번째 subject의 mrk가 다름
        if agent==1
            % Propofol         
            baseline_mrk= -(state==2 && i==1)+2; % 조건에 맞으면 1, 아니면 2
        else
            % Midazolam
            baseline_mrk=2;
        end
        
        
        for i=1:n
            tic;
            name=strcat(Type_filename{agent,state},num2str(i));

            [cnt, mrk, mnt]=file_loadMatlab([basic_dir name]); % Load cnt, mrk, mnt variables to Matlab

         
            count=0;
            for j=1:size(mrk.event.desc)
                if mrk.event.desc(j)==baseline_mrk
                    count=count+1;
                    X(:,:,count)=cnt.x(:,:,j);
                end
            end

            delta=[0.5 4]; theta=[4 8]; alpha=[8 15]; beta=[15 32];gamma=[32 60];
            temp=X(:,:,1:30);
            for j=1:30
                X_new(:,:,1,j)=bandpass(temp(:,:,j),delta,1000);
                X_new(:,:,2,j)=bandpass(temp(:,:,j),theta,1000);
                X_new(:,:,3,j)=bandpass(temp(:,:,j),alpha,1000);
                X_new(:,:,4,j)=bandpass(temp(:,:,j),beta,1000);
                X_new(:,:,5,j)=bandpass(temp(:,:,j),gamma,1000);
            end
            data.x.delta(:,:,:,i)=X_new(:,:,1,:);
            data.x.theta(:,:,:,i)=X_new(:,:,2,:);
            data.x.alpha(:,:,:,i)=X_new(:,:,3,:);
            data.x.beta(:,:,:,i)=X_new(:,:,4,:);
            data.x.gamma(:,:,:,i)=X_new(:,:,5,:);
            data.x.raw(:,:,:,i)=temp;
            
            general.weight(i)=cnt.weight;
            general.height(i)=cnt.height;
            general.age(i)=cnt.age;
            general.sex(i)=cnt.sex;
            
            fprintf('\n%s %d 번째 처리 시간: ',Type_filename{agent,state},i);
            toc;
        end

         title=strcat(...
             strcat('E:\data_result\',Type_filename{agent,state}),...
             'DataSet_All');

         save(title,'data','general','-v7.3');

    end
end
%% clustering 
function idx=clustering(x,k)

idx=kmeans(x,k);
idx(idx==2)=0;
if mean(x(idx==1))<mean(x(idx==0))
    idx=~idx; % 1이 Low sensitivity가 아니면 뒤집어줌 (Ce는 높아야함)
end
end
