function [new_cl,new_cr,new_ct] =anc_ch5(ll,rr,tr,n_ec1,n_ec2,n_ec3,ch,V,n)
for trial=1:ll
    primary=n_ec1(trial,:,ch);
    r=0;
    
    for k=1:28
        if V(ch,k)==1
            r=r+1;
            refer{r}(1,:)=n_ec1(trial,:,k);
        end
    end
    order=3;
    mu=0.0001;
    %     n=350;
    for k=1:r
        delayed{k}=zeros(1,order);
        adap{k}=zeros(1,order);
    end
    %%
    for k=1:1
        for kk=1:r
            delayed{kk}(3)=refer{kk}(k);
            y{kk}=delayed{kk}*adap{kk}';
        end
        temp=0;
        for kkk=1:r
            temp=temp+y{kkk};
        end
        if k==1
            temp=0;
        end
        cancelled(k)=primary(k)-temp;
%         mu=0.001./(0.01+delayed{kk}*delayed{kk}');
        for kk=1:r
            adap{kk} = adap{kk} + mu*cancelled(k) .* delayed{kk};
            %                         delayed{kk}(2:order)=delayed{kk}(1:order-1);
        end
    end
    %%
        for k=2:2
        for kk=1:r
            delayed{kk}(2:3)=refer{kk}(k-1:k);
            y{kk}=delayed{kk}*adap{kk}';
        end
        temp=0;
        for kkk=1:r
            temp=temp+y{kkk};
        end
        if k==1
            temp=0;
        end
        cancelled(k)=primary(k)-temp;
%         mu=0.001./(0.01+delayed{kk}*delayed{kk}');
        for kk=1:r
            adap{kk} = adap{kk} +mu*cancelled(k) .* delayed{kk};
            %                         delayed{kk}(2:order)=delayed{kk}(1:order-1);
        end
    end
    %%
    for k=3:n
        for kk=1:r
            delayed{kk}(1:3)=refer{kk}(k-2:k);
            y{kk}=delayed{kk}*adap{kk}';
        end
        temp=0;
        for kkk=1:r
            temp=temp+y{kkk};
        end
        if k==1
            temp=0;
        end
        cancelled(k)=primary(k)-temp;
%         mu=0.001./(0.01+delayed{kk}*delayed{kk}');
        for kk=1:r
            adap{kk} = adap{kk} + mu*cancelled(k) .* delayed{kk};
%             delayed{kk}(2:order)=delayed{kk}(1:order-1);
        end
    end
    new_cl(trial,:)=cancelled;
end

for trial=1:rr
    pprimary=n_ec2(trial,:,ch);
    r=0;
    for k=1:28
        if V(ch,k)==1
            r=r+1;
            rrefer{r}(1,:)=n_ec2(trial,:,k);
        end
    end
    %     order=3;
    %     mu=0.0001;
    %     n=300;
    for k=1:r
        delayed{k}=zeros(1,order);
        adap{k}=zeros(1,order);
    end
       %%
    for k=1:1
        for kk=1:r
            delayed{kk}(3)=rrefer{kk}(k);
            yy{kk}=delayed{kk}*adap{kk}';
        end
        temp=0;
        for kkk=1:r
            temp=temp+yy{kkk};
        end
        if k==1
            temp=0;
        end
        ccancelled(k)=pprimary(k)-temp;
%         mu=0.001./(0.01+delayed{kk}*delayed{kk}');
        for kk=1:r
            adap{kk} = adap{kk} + mu*ccancelled(k) .* delayed{kk};
            %                         delayed{kk}(2:order)=delayed{kk}(1:order-1);
        end
    end
    %%
        for k=2:2
        for kk=1:r
            delayed{kk}(2:3)=rrefer{kk}(k-1:k);
            yy{kk}=delayed{kk}*adap{kk}';
        end
        temp=0;
        for kkk=1:r
            temp=temp+yy{kkk};
        end
        if k==1
            temp=0;
        end
        ccancelled(k)=pprimary(k)-temp;
%         mu=0.001./(0.01+delayed{kk}*delayed{kk}');
        for kk=1:r
            adap{kk} = adap{kk} + mu*ccancelled(k) .* delayed{kk};
            %                         delayed{kk}(2:order)=delayed{kk}(1:order-1);
        end
    end
    %%
    for k=3:n
        for kk=1:r
            delayed{kk}(1:3)=rrefer{kk}(k-2:k);
            yy{kk}=delayed{kk}*adap{kk}';
        end
        temp=0;
        for kkk=1:r
            temp=temp+yy{kkk};
        end
        if k==1
            temp=0;
        end
        ccancelled(k)=pprimary(k)-temp;
%        mu=0.001./(0.01+delayed{kk}*delayed{kk}');
        for kk=1:r
            adap{kk} = adap{kk} + mu*ccancelled(k) .* delayed{kk};
%             delayed{kk}(2:order)=delayed{kk}(1:order-1);
        end
    end
    new_cr(trial,:)=ccancelled;
end

for trial=1:tr
    ppprimary=n_ec3(trial,:,ch);
    r=0;
    for k=1:28
        if V(ch,k)==1
            r=r+1;
            rrrefer{r}(1,:)=n_ec3(trial,:,k);
        end
    end
    %     order=3;
    %     mu=0.0001;
    %     n=300;
    for k=1:r
        delayed{k}=zeros(1,order);
        adap{k}=zeros(1,order);
    end
        %%
    for k=1:1
        for kk=1:r
            delayed{kk}(3)=rrrefer{kk}(k);
            yyy{kk}=delayed{kk}*adap{kk}';
        end
        temp=0;
        for kkk=1:r
            temp=temp+yyy{kkk};
        end
        if k==1
            temp=0;
        end
        cccancelled(k)=ppprimary(k)-temp;
%         mu=0.001./(0.01+delayed{kk}*delayed{kk}');
        for kk=1:r
            adap{kk} = adap{kk} + mu*cccancelled(k) .* delayed{kk};
            %                         delayed{kk}(2:order)=delayed{kk}(1:order-1);
        end
    end
    %%
        for k=2:2
        for kk=1:r
            delayed{kk}(2:3)=rrrefer{kk}(k-1:k);
            yyy{kk}=delayed{kk}*adap{kk}';
        end
        temp=0;
        for kkk=1:r
            temp=temp+yyy{kkk};
        end
        if k==1
            temp=0;
        end
        cccancelled(k)=ppprimary(k)-temp;
%          mu=0.001./(0.01+delayed{kk}*delayed{kk}');
        for kk=1:r
            adap{kk} = adap{kk} + mu*cccancelled(k) .* delayed{kk};
            %                         delayed{kk}(2:order)=delayed{kk}(1:order-1);
        end
        end
    %%
    for k=3:n
        for kk=1:r
            delayed{kk}(1:3)=rrrefer{kk}(k-2:k);
            yyy{kk}=delayed{kk}*adap{kk}';
        end
        temp=0;
        for kkk=1:r
            temp=temp+yyy{kkk};
        end
        if k==1
            temp=0;
        end
        cccancelled(k)=ppprimary(k)-temp;
%        mu=0.001./(0.01+delayed{kk}*delayed{kk}');
        for kk=1:r
            adap{kk} = adap{kk} + mu*cccancelled(k) .* delayed{kk};
%             delayed{kk}(2:order)=delayed{kk}(1:order-1);
        end
    end
    new_ct(trial,:)=cccancelled;
end