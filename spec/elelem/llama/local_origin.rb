# frozen_string_literal: true

require "socket"

class LocalOrigin
  Reply = Data.define(:status, :headers, :body, :length)

  attr_reader :requests

  def self.reply(status = "200 OK", body: "", headers: {}, length: body.bytesize)
    Reply.new(status:, headers:, body:, length:)
  end

  def initialize(&handler)
    @handler = handler
    @requests = []
    @server = TCPServer.new("127.0.0.1", 0)
    @thread = Thread.new { serve }
  end

  def uri = URI("http://127.0.0.1:#{@server.addr[1]}")

  def close
    @thread.kill
    @server.close
  end

  private

  def serve
    loop do
      client = @server.accept
      path = client.gets.split[1]
      headers = client.each_line("\r\n", chomp: true).take_while { !it.empty? }.to_h { it.split(": ", 2) }
      @requests << [path, headers.transform_keys(&:downcase)]
      write(client, @handler.call(path, @requests.last.last))
    end
  end

  def write(client, reply)
    client.write("HTTP/1.1 #{reply.status}\r\n")
    { "Content-Length" => reply.length, "Connection" => "close", **reply.headers }.each do |name, value|
      client.write("#{name}: #{value}\r\n")
    end
    client.write("\r\n#{reply.body}")
    client.close
  rescue SystemCallError
    client.close
  end
end
