#!/usr/bin/env python

import os
import sys
import time
import glob
import shutil
import subprocess
from daemon import Daemon
from operator import itemgetter

# define project directory location
# TODO add a set_root_dir function, or pass spica as parameter...
project_dir = '/home/bastiaan/Develop/spiceweb_test/projects'

# define log output files
log_dir = '/home/bastiaan/Develop/spice/spice/job_runner'
pid_f = os.path.join(log_dir, 'daemon.pid')

# pause between checking for jobs in queue
sleep_interval = 5


class JobRunner(Daemon):

    def run(self):
        while True:

            # obtain job list (jobf, project_dir, timestamp)
            jobs = []
            for project_d in glob.glob(os.path.join(project_dir, '*', '*')):

                # check for files in the waiting directories
                wait_f = os.path.join(project_d, 'jobs', 'waiting', '*')
                for f in glob.glob(wait_f):
                    job_f = os.path.basename(f)
                    timestamp = int(''.join(job_f.split('_')[:1]))
                    with open(f, 'r') as fin:
                        cmd = '%s' % (fin.readline())
                        stdout_f = fin.readline().strip()
                        stderr_f = fin.readline().strip()
                    jobs.append((project_d, job_f, cmd, timestamp))

            # if found, run the one that was submitted first
            if(len(jobs) > 0):

                # sort by submitted time
                (project_d, job_f, cmd, t) = sorted(jobs, key=itemgetter(3))[0]

                # define file paths
                wait_f = os.path.join(project_d, 'jobs', 'waiting', job_f)
                run_f = os.path.join(project_d, 'jobs', 'running', job_f)
                done_d = os.path.join(project_d, 'jobs', 'done')
                done_f = os.path.join(done_d, job_f)
                err_d = os.path.join(project_d, 'jobs', 'error')
                err_f = os.path.join(err_d, job_f)

                # move job file to running dir
                shutil.move(wait_f, run_f)

                # redirect standard file descriptors
                sys.stdout.flush()
                sys.stderr.flush()
                self.stdout = stdout_f
                self.stderr = stderr_f
                so = file(self.stdout, 'a+')
                se = file(self.stderr, 'a+', 0)
                os.dup2(so.fileno(), sys.stdout.fileno())
                os.dup2(se.fileno(), sys.stderr.fileno())

                # run the job (in a later stage thread the job...? Popen)
                retcode = subprocess.call(cmd.split())

                # move running to either done or error based on return code
                if(retcode == 0):
                    shutil.move(run_f, done_f)
                else:
                    shutil.move(run_f, err_f)

                # move results...

            # then sleep for a while before going to the next loop
            time.sleep(sleep_interval)


if __name__ == "__main__":

    # create the daemon
    daemon = JobRunner(pid_f)

    # start, stop, or restart the jobrunner daemon
    if len(sys.argv) == 2:
        if 'start' == sys.argv[1]:
            daemon.start()
        elif 'stop' == sys.argv[1]:
            daemon.stop()
        elif 'restart' == sys.argv[1]:
            daemon.restart()
        else:
            print "Unknown command"
            sys.exit(2)
        sys.exit(0)
    else:
        print "usage: %s start|stop|restart" % sys.argv[0]
        sys.exit(2)
