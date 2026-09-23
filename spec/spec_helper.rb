# frozen_string_literal: true

require "fileutils"
require "tmpdir"
require_relative "../lib/elelem/llama/model_resolver"

RSpec.configure do |config|
  config.disable_monkey_patching!

  config.expect_with :rspec do |c|
    c.syntax = :expect
  end
end
