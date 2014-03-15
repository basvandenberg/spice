#!/usr/bin/env python

import os
import time
import glob
import shutil
import subprocess
from daemon import Daemon
from operator import itemgetter

job_types = ['featext', 'classification', 'classify']
max_num_jobs = [1, 1, 1]

# maximal allowed number of jobs per job type
max_jobs = dict(zip(job_types, max_num_jobs))

# pause between checking for jobs in queue
sleep_interval = 5


class JobQueueManager(Daemon):

    def __init__(self, pidfile, project_dir, stdin='/dev/null',
                 stdout='/dev/null', stderr='/dev/null'):

        super(JobQueueManager, self).__init__(pidfile, stdin, stdout, stderr)

        # somehow a print is required here, to get output for the stdout...
        # HACK, don't remove!
        print('')

        # projects directory
        self.project_dir = os.path.abspath(project_dir)

        # list of running jobs per job type, to keep track of all running jobs
        self.running_jobs = dict(zip(
            job_types, [[] for i in xrange(len(job_types))]))

    def run(self):

        while True:

            # remove finished jobs from running list
            for job_type in job_types:
                for job_tuple in self.running_jobs[job_type]:

                    job, run_f, done_f, err_f, fout, ferr = job_tuple

                    # close the log files # WHY IS THIS HERE???
                    fout.close()
                    ferr.close()

                    # check if it is finished
                    job_status = job.poll()
                    if not(job.poll() is None):

                        # remove the job from the list
                        self.running_jobs[job_type].remove(job_tuple)

                        if(job_status == 0):
                            shutil.move(run_f, done_f)
                        else:
                            shutil.move(run_f, err_f)

            # obtain job lists
            # {jobtype: [(project_dir, job_f, cmd, t, stdout_f, stderr_f)]}
            jobs = dict(zip(job_types, [[] for i in xrange(len(job_types))]))

            for project_d in glob.glob(
                    os.path.join(self.project_dir, '*', '*')):

                # check for files in the waiting directories
                wait_f = os.path.join(project_d, 'jobs', 'waiting', '*')
                for f in glob.glob(wait_f):
                    job_f = os.path.basename(f)
                    t = int(''.join(job_f.split('_')[:2]))
                    with open(f, 'r') as fin:
                        cmd = '%s' % (fin.readline())
                        job_type = cmd.split()[0]
                        job_stdout_f = fin.readline().strip()
                        job_stderr_f = fin.readline().strip()

                    jobs.setdefault(job_type, [])\
                        .append((project_d, job_f, cmd, t,
                                 job_stdout_f, job_stderr_f))

            # for each job type
            for job_type in job_types:

                # check if max number of jobs are running for this job type
                if(len(jobs[job_type]) > 0 and
                   len(self.running_jobs[job_type]) < max_jobs[job_type]):

                    # sort jobs current job type by timestamp and select first
                    first_job = sorted(jobs[job_type], key=itemgetter(3))[0]
                    project_d, job_f, cmd, _, jobout_f, joberr_f = first_job

                    # define file paths
                    wait_f = os.path.join(project_d, 'jobs', 'waiting', job_f)
                    run_f = os.path.join(project_d, 'jobs', 'running', job_f)
                    done_f = os.path.join(project_d, 'jobs', 'done', job_f)
                    err_f = os.path.join(project_d, 'jobs', 'error', job_f)

                    fout = open(jobout_f, 'w')
                    ferr = open(joberr_f, 'w')
                    job = subprocess.Popen(cmd, shell=True, stdout=fout,
                                           stderr=ferr)

                    # move job file from waiting to running
                    shutil.move(wait_f, run_f)

                    # add job to running jobs list
                    self.running_jobs[job_type].append(
                        (job, run_f, done_f, err_f, fout, ferr))

            '''
            print
            print 'Queue:'
            print jobs
            print
            print 'Running:'
            print self.running_jobs
            print
            '''

            # then sleep for a while before going to the next loop
            time.sleep(sleep_interval)


#if __name__ == "__main__":
# TODO add test runs
