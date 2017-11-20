function [map_out]=Model_Map_update(myMap,case_update,opt)


% 'Tprob' : transition probability update
% Tprob{1,L|R}: column-L/R in next state, row-each current state
% Tprob{1,L|R}(i,:): probability set of being taken to the (left, right) state when making L choice in i-th state
% e.g., Tprob{1,1}(4,:) = [0.7 0.3] : transition prob. to the (left,right) state when making L choice in state 4.
% e.g., Tpro{1,2}(4,:) = [0.7 0.3] : transition prob. to the (left,right) state when making R choice in state 4.

% 'Rwd' : reward value update

N_state=myMap.N_state;

switch lower(myMap.map_name)
    case {'sangwan2012c'},
        
        switch case_update
            case {'T'} % transition probability update
                Tprob=opt;
                for mm=1:1:2 % for left/right action case
                    % for non-zero connections only
                    row2visit=find(sum(myMap.connection_info{1,mm},2)~=0);
                    for nn=row2visit' % visit valid connections and update
                        myMap.action(1,mm).prob(nn,myMap.connection_info{1,mm}(nn,:))=Tprob;
                    end
                    % 3. get a connection matrix
                    myMap.action(1,mm).connection=double(myMap.action(1,mm).prob&ones(N_state,N_state));
                end

            case {'R'} % reward value update
                Rwd_condition=opt;
                
                switch Rwd_condition
                    case -1
                        myMap.reward(6:8)=[40 20 10];
                    case 6
                        myMap.reward(6:8)=[40 0 0];
                    case 7
                        myMap.reward(6:8)=[0 20 0];                    
                    case 8
                        myMap.reward(6:8)=[0 0 10];       
                end
                
        end        
    
    

    % old version - not using anymore
    case {'sangwan2014a'},

        switch case_update

            case {'T'} % transition probability update
                Tprob=opt;
                for mm=1:1:2 % for left/right action case
                    row2visit=find(sum(myMap.connection_info{1,mm},2)~=0);
                    for nn=row2visit' % visit valid connections and update
                        myMap.action(1,mm).prob(nn,myMap.connection_info{1,mm}(nn,:))=Tprob{1,mm}(nn,:);
                    end
                    % 3. get a connection matrix
                    myMap.action(1,mm).connection=double(myMap.action(1,mm).prob&ones(N_state,N_state));
                end

            case {'R'} % reward value update
                Rwd_array=opt;
                for rr=1:1:size(myMap.goal_state_index,2)
                    myMap.reward(myMap.goal_state_index{1,rr})=Rwd_array(rr);
                end

        end


    % new version
    case {'sangwan2014b'},

        switch case_update

            case {'T'} % transition probability update
                Tprob=opt;
                for mm=1:1:4 % for left/right action case
                    % for non-zero connections only
                    row2visit=find(sum(myMap.connection_info{1,mm},2)~=0);
                    for nn=row2visit' % visit valid connections and update
                        myMap.action(1,mm).prob(nn,myMap.connection_info{1,mm}(nn,:))=Tprob;
                    end
                    % 3. get a connection matrix
                    myMap.action(1,mm).connection=double(myMap.action(1,mm).prob&ones(N_state,N_state));
                end

            case {'R'} % reward value update
                Rwd_array=opt;
                for rr=1:1:size(myMap.goal_state_index,2) % 항상 3으로 고정
                    myMap.reward(myMap.goal_state_index{1,rr})=Rwd_array(rr);
                end

        end

    otherwise,
        error('this map type does not support transition probability change!');

end

map_out=myMap;

end