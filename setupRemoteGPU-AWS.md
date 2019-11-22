### Setting up remote GPU AWS ###

1. On the AWS instance dashboard select the gpu and launch the instance, click the connect tab above for
connect details - Public DNS and ssh details

2. chmod 400 the .pem file

3. ssh into the GPU instance using the given command from the connect tab - something like this
ssh -i PATH_TO_PEM_FILE ubuntu@INSTANCE_IP_ADDRESS

4. edit the jupyter config file file on the instance
nano ~/.jupyter/jupyter_notebook_config.py

and insert the following lines at the top

c = get_config()
c.IPKernelApp.pylab = 'inline'
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8888
c.NotebookApp.token = ''
c.NotebookApp.password = ''

the jupyter notebook will now launch on the gpu instance at port 8888

5. exit the ssh using crtl+d

6. foward a port on your local machine to view the jupyter notebook on the remote gpu
something like this witht he same details as in step 3
ssh -i PATH_TO_PEM_FILE -fNL 9000:localhost:8888 ubuntu@INSTANCE_IP_ADDRESS

9000 is now the local port you can view the remote jupyter notebook on

7. Open up your browser and enter: http://localhost:8888
to view the remote jupoyter notebook.