% PSD_modeling.m
%
%
%
% author: Young-Seok, Kweon
% created: 2019.10.15
%% init

clc; clear; close all;

%% load 

type={'HP','MP','LP'};
n_type=size(type,2);
n=10;

g= @(p1,p2,k1,k2,u1,u2,s1,s2,lambda,x)...
    p1+p2./power(x,lambda)...
    +k1*normpdf(x,u1,s1)+k2*normpdf(x,u2,s2);
f=[0.5:0.5:40];
for i=1:n_type
    for j=1:n
        x=load(['Feature_PSD\',type{i},'_S',num2str(j),'_PSD']);
        data=x.data;
        label=x.label;
        frontal_base=mean(data(1:62,:,label==0),3);
        for ch=1:size(frontal_base,1)
            trained_g=fit(f',frontal_base(ch,1:80)',g,...
                'StartPoint',[0,0,1,1,10,20,1,1,1],...
                'Upper',[inf,inf,inf,inf,15,30,100,100,100],...
                'Lower',[-inf,-inf,0,0,0,15,0,0,0]);
            g1=trained_g.p1+trained_g.p2./power(f,trained_g.lambda);
            y=trained_g(f);
            [v,idx]=maxk(y-g1',1);
            predictor(i,j,ch)=v;
            k1(i,j,ch)=trained_g.k1;
            k2(i,j,ch)=trained_g.k2;
        end
        fprintf('%s [done]\n',[type{i},'_S',num2str(j)]);
    end
end
p=[];k1_t=[];k2_t=[];
for i=1:n_type
    p=cat(1,p,reshape(predictor(i,:,:),size(predictor,2),size(predictor,3)));
    k1_t=cat(1,k1_t,reshape(k1(i,:,:),size(k1,2),size(k1,3)));
    k2_t=cat(1,k2_t,reshape(k2(i,:,:),size(k2,2),size(k2,3)));
end
file=fopen('result.txt','w');
for i=1:size(p,1)
    for j=1:size(p,2)
        fprintf(file,'%f\t',p(i,j));
    end
    fprintf(file,'\n');
end
fprintf(file,'\n');
p=k1_t;
for i=1:size(p,1)
    for j=1:size(p,2)
        fprintf(file,'%f\t',p(i,j));
    end
    fprintf(file,'\n');
end
fprintf(file,'\n');
p=k2_t;
for i=1:size(p,1)
    for j=1:size(p,2)
        fprintf(file,'%f\t',p(i,j));
    end
    fprintf(file,'\n');
end