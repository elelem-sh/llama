# frozen_string_literal: true

require "bundler/gem_tasks"

desc "Build the in-process llama.cpp extension (the same build `gem install` runs)"
task :compile do
  ruby "ext/elelem/llama/extconf.rb"
end

task default: %i[]
