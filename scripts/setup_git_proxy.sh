#/bin/bash

if ping -c 1 -w 1 192.168.0.91 
then
    export http_proxy=http://192.168.0.91:12798 && export https_proxy=http://192.168.0.91:12798
    echo 'proxy setup at 192.168.0.91 WUHU'
else
    if ping -c 1 -w 1 100.72.64.19
    then
        export http_proxy=http://100.72.64.19:12798 && export https_proxy=http://100.72.64.19:12798
        echo 'proxy setup at 100.72.64.19 BEIJING'
    else
        if ping -c 1 -w 1 192.168.1.174
        then
            export http_proxy=http://192.168.1.174:12798 && export https_proxy=http://192.168.1.174:12798
            echo 'proxy setup at 92.168.1.174 NEIMENG'
        else
            if ping -c 1 -w 1 10.55.146.88 
            then
                export http_proxy=http://10.55.146.88:12798 && export https_proxy=http://10.55.146.88:12798
                echo 'proxy setup at 10.55.146.88 QUANZHOU'
            else
                if ping -c 1 -w 1 172.181.217.43 
                then
                    export http_proxy=http://172.181.217.43:12798 && export https_proxy=http://172.181.217.43:12798
                    echo 'proxy setup at 172.181.217.43 NANJING'
                else
                    echo 'no proxy found'
                fi
            fi
        fi
    fi
fi



