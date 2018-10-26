function bestlabel = knn(qn,fn,datastore)

    % sort the features according to dot product (i.e. a similarity)
    ss = fn'*qn;
    [s,y] = sort(ss,'descend');

    % count unique labels
    lc = countEachLabel(datastore);
    labels = lc(:,1);
    k = size(lc,1);
    
    % convert the table to simple array of strings
    labs = string(table2array(labels));
    alllabels = string(datastore.Labels);
    
    % hashmap for counting the labels
    Mcount = containers.Map(labs,zeros(1,size(labels,1)));
    
    % Confindence hashmap
    Mconf = containers.Map(labs,zeros(1,size(labels,1)));
    
    % take the winner 'label' among the first k
    for i=1:k
        Mcount(alllabels(y(i))) = Mcount(alllabels(y(i))) + 1;
        % set the confidence as the best similarity of the corresponding label
        if (Mconf(alllabels(y(i))) == 0)
           Mconf(alllabels(y(i))) = s(i);
        end
    end

    % get the best
    [mi,bi] = max(cell2mat(Mcount.values));
    
    % return the label
    % if the confidence is lower than 0.5 it returns 'unknow'
    if (Mconf(labs(bi)) > 0.5)
        bestlabel = labs(bi);
    else
        bestlabel = 'Unknow';
    end
end

